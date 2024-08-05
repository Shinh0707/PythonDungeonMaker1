from collections import deque
import heapq
import os
import matplotlib.animation as animation
import random
from typing import Callable, Literal
import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray, zeros_like
import pygame
from scipy import signal
from tqdm import tqdm

class Constant:
    def __init__(self) -> None:
        RotMasks = []
        S1 = np.array(
            [[1, 0, 1],
             [0, 0, 1],
             [1, 1, 1]],
            dtype=np.bool_
        )
        S2 = np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [1, 1, 1]],
            dtype=np.bool_
        )
        RotMasks.append(S1)
        RotMasks.append(S2)
        for _ in range(3):
            S1 = np.rot90(S1, k=-1)
            S2 = np.rot90(S2, k=-1)
            RotMasks.append(S1)
            RotMasks.append(S2)
        self.RotMasks = np.stack(RotMasks)
        self.RotLabel = np.array(
            [[0, 1, 2],
             [7, 8, 3],
             [6, 5, 4]]
        )
        self.surround_mask = np.array(
            [[1,1,1],
             [1,0,1],
             [1,1,1]],
            dtype=np.bool_
        )
        self.neighbor_mask = np.array(
            [[0,1,0],
             [1,0,1],
             [0,1,0]],
            dtype=np.bool_
        )
    
    @classmethod
    def find_random_position(cls, field: ndarray, search_value: int = 0, mask: ndarray|None = None) -> tuple[int, int]:
        """
        field内でsearch_valueと一致する値のランダムな位置を返します。

        Args:
        field (np.ndarray): 検索対象の2次元NumPy配列
        search_value (int): 検索する値

        Returns:
        tuple[int, int]: ランダムに選ばれた位置 (行, 列)。
                        該当する値が見つからない場合は None を返します。
        """
        # search_valueと一致する位置のインデックスを取得
        matching_positions = np.argwhere(
            field == search_value) if mask is None else np.argwhere((field == search_value) & mask)

        # 一致する位置がない場合はNoneを返す
        if matching_positions.size == 0:
            return None

        # ランダムに1つの位置を選択
        random_index = np.random.randint(matching_positions.shape[0])
        selected_position = tuple(matching_positions[random_index])

        return selected_position
    
    @classmethod
    def get_area_mask(cls, route_labels: ndarray, labels: ndarray, mode: Literal['max', 'min', 'random'] = 'max'):
        assert mode in ['max', 'min', 'random']
        area_sizes = [np.sum(labels == rl) for rl in route_labels]
        if mode == 'max':
            selected_label = route_labels[np.argmax(area_sizes)]
        elif mode == 'min':
            selected_label = route_labels[np.argmin(area_sizes)]
        else:
            selected_label = route_labels[random.randint(0,len(route_labels)-1)]
        return labels == selected_label, selected_label
    
    @classmethod
    def extract_surrounding_efficient(cls, pos: tuple[int, int], field: ndarray, return_generator: bool = False, pad_value: int = 1):
        """
        指定された位置(pos)を中心とする3x3の領域をfieldから効率的に抽出します。

        Args:
        pos (tuple[int, int]): 中心位置の(行, 列)座標
        field (np.ndarray): 2次元のnumpy配列
        pad_value (int): 境界外を埋める値 default 1

        Returns:
        np.ndarray: 3x3の抽出された領域。境界外の場合はパディングとしてpad_valueが使用されます。
        """
        row, col = pos
        rows, cols = field.shape

        # 抽出する領域の範囲を計算
        row_start = max(0, row - 1)
        row_end = min(rows, row + 2)
        col_start = max(0, col - 1)
        col_end = min(cols, col + 2)

        def generator(mat):
            # 実際のフィールドから領域を抽出
            extracted = mat[row_start:row_end, col_start:col_end]

            # 結果を格納する3x3の配列を作成
            result = np.full((3, 3), pad_value)

            # 抽出した領域を結果配列の適切な位置に配置
            result_start_row = 1 - (row - row_start)
            result_start_col = 1 - (col - col_start)
            result[result_start_row:result_start_row+extracted.shape[0],
                result_start_col:result_start_col+extracted.shape[1]] = extracted
            return result
        
        if return_generator:
            return generator(field), generator
        return generator(field)
    
    def is_potential_splitter(self, extracted_field: ndarray) -> bool:
        if extracted_field[1,1] == 1:
            return False
        field_mask = (extracted_field == 1) & self.surround_mask
        if not field_mask.any():
            return False
        return np.sum(field_mask & (np.sum(self.RotMasks[self.RotLabel[field_mask]], axis=0)>0)) > 0
    
    @classmethod
    def get_labels(cls, field: ndarray):
        retval, labels = cv2.connectedComponents(
            np.array(1-field, dtype=np.int8), connectivity=4)
        wall_labels = np.argwhere(field == 1)
        wall_label = (-1) if wall_labels.size == 0 else labels[*wall_labels[0]].item()
        route_labels = np.array(
            [rval for rval in range(retval) if rval != wall_label])
        return route_labels, labels, wall_label
    
    def set_wall(self, pos: tuple[int, int], field: ndarray, min_size: int, route_labels: ndarray | None = None, labels: ndarray | None = None):
        if labels is None:
            route_labels, labels, _ = self.get_labels(field)
        elif route_labels is None:
            route_labels = np.unique(labels[field == 0])
        if field[*pos] == 1:
            return True, field, route_labels, labels
        if np.sum(labels == labels[*pos].item()) - 1 < min_size:
            return False, field, route_labels, labels
        extracted_field, generator = self.extract_surrounding_efficient(
            pos, field, return_generator=True)
        pred_field = field.copy()
        pred_field[*pos] = 1
        pred_field_labels = labels.copy()
        wall_labels = np.argwhere(field == 1)
        if wall_labels.size == 0:
            pred_route_labels, pred_field_labels, _ = self.get_labels(
                pred_field)
            return True, pred_field, pred_route_labels, pred_field_labels
        wall_label = pred_field_labels[*wall_labels[0]].item()
        pred_field_labels[*pos] = wall_label
        pred_route_labels = route_labels.copy()
        if self.is_potential_splitter(extracted_field):
            pred_route_labels, pred_field_labels, wall_label = self.get_labels(
                pred_field)
            # print(f"Try to Update Labels")
            for ul in pred_route_labels:
                if np.sum(pred_field_labels == ul) < min_size:
                    # print(f"Failed Update Labels")
                    return False, field, route_labels, labels
        return True, pred_field, pred_route_labels, pred_field_labels
    
    def delete_wall(self, pos: tuple[int, int], field: ndarray, route_labels: ndarray | None = None, labels: ndarray | None = None):
        if labels is None:
            route_labels, labels, _ = self.get_labels(field)
        elif route_labels is None:
            route_labels = np.unique(labels[field == 0])
        if field[*pos] == 0:
            return True, field, route_labels, labels
        extracted_field, generator = self.extract_surrounding_efficient(
            pos, field, return_generator=True)
        mask = self.neighbor_mask & (extracted_field == 0)
        if not mask.any():
            return False, field, route_labels, labels
        pred_field_labels = labels.copy()
        uniq_labels = np.unique(generator(labels)[mask])
        wall_label = labels[*np.argwhere(field == 1)[0]].item()
        uniq_labels = uniq_labels[uniq_labels != wall_label]
        min_label = np.min(uniq_labels)
        del_labels = uniq_labels[uniq_labels != min_label]
        pred_route_labels = np.array(
              [rl for rl in route_labels if not rl in del_labels])
        for dl in del_labels:
            # print(f"Delete Label {dl}")
            pred_field_labels[pred_field_labels == dl] = min_label
        pred_field_labels[*pos] = min_label
        pred_field = field.copy()
        pred_field[*pos] = 0
        return True, pred_field, pred_route_labels, pred_field_labels
    
    def auto_set(self, field: ndarray, value: int, min_size: int, route_labels: ndarray | None = None, labels: ndarray | None = None):
        if labels is None:
            route_labels, labels, _ = self.get_labels(field)
        elif route_labels is None:
            route_labels = np.unique(labels[field == 0])
        if value == 0:
            field_mask = field == 1
        else:
            field_mask, selected_label = self.get_area_mask(route_labels,labels,mode='max')
            # print(f"Size is {np.sum(labels == selected_label).item()}")
        pos = self.find_random_position(field, 1-value, field_mask)
        if pos is None:
            return field, route_labels, labels
        if value == 0:
            # print(f"delete wall")
            return self.delete_wall(pos, field, route_labels, labels)[1:]
        # print(f"set wall")
        return self.set_wall(pos, field, min_size, route_labels, labels)[1:]
    
    def auto_setting(self, field: ndarray, min_size: int, count: int=100, route_labels: ndarray | None = None, labels: ndarray | None = None):
        if labels is None:
            route_labels, labels, _ = self.get_labels(field)
        elif route_labels is None:
            route_labels = np.unique(labels[field == 0])
        hist = [(field, route_labels, labels)]
        for _ in tqdm(range(count), desc="Auto Setting Progress", ncols=100):
            hist.append(self.auto_set(
                hist[-1][0], random.sample([0, 1], k=1, counts=[1, 10])[0], min_size, hist[-1][1], hist[-1][2]))
        r, l, _ = self.get_labels(hist[-1][0])
        hist.append((hist[-1][0], r, l))
        return hist
    
    @classmethod
    def get_peak2D(cls, A: np.ndarray, mask: np.ndarray = None, mode: Literal['maximum', 'minimum'] = 'maximum'):
        if mask is not None:
            A = np.where(mask, A, np.min(A))

        pad_mode = 'minimum'

        # y方向の処理
        dy = np.diff(np.pad(A, ((1, 1), (0, 0)), mode=pad_mode), axis=0)
        P_dy = np.diff(np.sign(dy), axis=0)

        # x方向の処理
        dx = np.diff(np.pad(A, ((0, 0), (1, 1)), mode=pad_mode), axis=1)
        P_dx = np.diff(np.sign(dx), axis=1)

        if mode == 'maximum':
            Res = (P_dy == -2) & (P_dx == -2)
        else:  # minimum
            Res = (P_dy == 2) & (P_dx == 2)

        if mask is not None:
            Res &= mask

        return Res
    
    @classmethod
    def RN2C(cls, x: ndarray):
        """
        Range Normalization with Double Centering
        データに対して中心化、最小-最大スケーリング、[-1, 1]範囲への変換、そして再度の中心化を適用する
        """
        x_centered = 2 * cls.min_max_normalize(x - np.mean(x)) - 1
        return x_centered - np.mean(x_centered)
    
    @classmethod
    def min_max_normalize(cls, x:ndarray, if_zero_value: float=1.0):
        reduced_x = x - np.min(x)
        max_x = np.max(reduced_x)
        if max_x == 0.0:
            return np.full_like(x,if_zero_value)
        return reduced_x/max_x
    
class Analyzer:
    @classmethod
    def neighbor_count(cls, field: ndarray):
        kernel = np.array(
            [[0,1,0],
             [1,0,1],
             [0,1,0]]
        )
        padded_field = np.pad(1-field, 1, mode='constant', constant_values=0)
        mask = field == 0
        R = zeros_like(field)
        R[mask] = signal.convolve2d(padded_field, kernel, mode='valid')[mask]
        return R
    
    @classmethod
    def neighbor_score(cls, field: ndarray):
        return cls.neighbor_count(field)

    @classmethod
    def delta_score(cls, field: ndarray, neighbor_score: ndarray|None=None):
        if neighbor_score is None:
            neighbor_score = cls.neighbor_score(field)
        kernel = np.array(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]]
        )
        mask = field == 0
        R = np.zeros_like(field, dtype=float)

        #padded_field = np.pad(field, 1, mode='constant', constant_values=1)
        padded_neighbor_score = np.pad(neighbor_score, 1, mode='constant', constant_values=0)

        #shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        #for di, dj in shifts:
        #    R += np.roll(np.roll(padded_neighbor_score, di, axis=0), dj, axis=1)[1:-1, 1:-1] * (
        #        np.roll(np.roll(padded_field, di, axis=0), dj, axis=1)[1:-1, 1:-1] != 1)

        #R -= 4 * neighbor_score
        R[mask] = signal.convolve2d(padded_neighbor_score,kernel,mode='valid')[mask]
        return R
    
    @classmethod
    def difficulty(cls, field: np.ndarray, alpha: float = 0.5, steps: int = 1000, neighbor_score: np.ndarray | None = None, delta_score: np.ndarray | None = None, target_labels:ndarray|None=None, labels: ndarray|None=None):
        if delta_score is None:
            delta_score = cls.delta_score(field, neighbor_score)

        neighbor_count = cls.neighbor_count(field)

        def norm(selector):
            delta_score[selector] = Constant.RN2C(delta_score[selector])

        if not (target_labels is None or labels is None):
            cls.apply_each_areas(target_labels,labels, lambda selector,lbl: norm(selector))

        dx = dy = 1.0
        dt = 1.0
        ddx = ddy = 1.0 / dx**2
        alpha = min(alpha, (0.5 / (ddx + ddy)) / dt)

        R = delta_score.copy()
        mask = field == 0

        # カーネルを作成
        kernel = np.array([
            [0, ddy, 0],
            [ddx, 0, ddx],
            [0, ddy, 0]]
        )

        kernel = alpha * dt * kernel
        neighbor_count = alpha * dt * neighbor_count * ddx

        hist = []
        for _ in tqdm(range(steps), desc="Difficulty Heat Diffusion in Progress", ncols=100):
            hist.append(R.copy())

            # 畳み込み演算
            laplacian = np.pad(R, 1, mode='constant', constant_values=0)
            laplacian = signal.convolve2d(laplacian, kernel, mode='valid')

            # 更新
            R[mask] += laplacian[mask] - neighbor_count[mask]*R[mask]
            R[~mask] = 0

        return R, hist
    
    @classmethod
    def difficulty_peaks(cls, field: ndarray, difficulty_score: ndarray | None = None, target_labels: ndarray | None = None, labels: ndarray | None = None,** kwargs):
        if difficulty_score is None:
            difficulty_score = cls.difficulty(
                field, target_labels=target_labels, labels=labels,**kwargs)[0]
        max_peaks = Constant.get_peak2D(
            difficulty_score, field == 0, 'maximum')
        min_peaks = Constant.get_peak2D(
            difficulty_score, field == 0, 'minimum')
        normalized_peak_value = np.zeros_like(field)
        
        def norm(selector):
            max_selector = selector & max_peaks
            min_selector = selector & min_peaks
            if max_selector.any():
                normalized_peak_value[max_selector] = Constant.min_max_normalize(
                    difficulty_score[max_selector]
                ) + 1
            else:
                max_selector = selector & (
                    difficulty_score == np.max(difficulty_score[selector]))
                normalized_peak_value[max_selector] = 2
            if min_selector.any():
                normalized_peak_value[min_selector] = -(Constant.min_max_normalize(
                    -difficulty_score[min_selector]
                ) + 1)
            else:
                min_selector = selector & (
                    difficulty_score == np.min(difficulty_score[selector]))
                normalized_peak_value[min_selector] = -2
        
        if target_labels is None or labels is None:
            norm(field == 0)
        else:
            cls.apply_each_areas(target_labels, labels,
                                 lambda selector, lbl: norm(selector))
        
        return max_peaks, min_peaks, normalized_peak_value
    
    @classmethod
    def apply_each_areas(cls, target_labels: ndarray, labels: ndarray, apply_func: Callable[[ndarray,int],None]):
        for lbl in target_labels:
            apply_func(labels==lbl,lbl)

    @classmethod
    def fluid(cls, sources: np.ndarray, stable: np.ndarray, mask: np.ndarray, steps: int,
              fmax: float, fmin: float, alpha: float = 0.5):

        h, w = sources.shape
        alpha = min(alpha, 1.0)
        if mask is None:
            mask = np.ones_like(sources, dtype=bool)

        # 方向カーネルを作成
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        # 有効なセルの数を計算
        count_table = np.sum(np.stack([np.roll(mask, (i, j), (0, 1)) for i, j in [
                            (0, 1), (0, -1), (1, 0), (-1, 0)]]), axis=0) * mask

        # 更新が必要なセルを特定
        updateds_checker = (stable != 0) & (count_table > 0)

        fluid_hist = []
        fluid_current = sources.copy()
        thres_fill = (np.mean(sources)*0.2 + np.min(sources)*0.8)
        filled_steps = np.full_like(sources,-1)

        for step in tqdm(range(steps), desc="Difficulty Fluid Diffusion in Progress", ncols=100):
            fluid_hist.append(fluid_current.copy())
            filled_steps[(filled_steps == -1) &
                         (fluid_hist[-1] > thres_fill)] = step
            if not ((filled_steps == -1) & mask).any():
                break


            # 近傍セルの値の合計を計算
            neighbor_sum = np.sum(np.stack([np.roll(fluid_hist[-1] * stable * mask, (i, j), (0, 1))
                                for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]]), axis=0)

            # D_copyを更新
            fluid_current += alpha * neighbor_sum / (count_table + 1) * updateds_checker
            fluid_current -= alpha * fluid_hist[-1] * stable * \
                (count_table / (count_table + 1)) * updateds_checker

            # 値の範囲を制限
            np.clip(fluid_current, fmin, fmax, out=fluid_current)

        filled_steps[(filled_steps == -1) & mask] = np.max(filled_steps) + 1

        return fluid_current, fluid_hist, filled_steps/np.max(filled_steps)
    
    @classmethod
    def fluid_difficulty(cls, field: ndarray, source_amount: float = 3.0, steps: int = 1000, normalized_peak_value: ndarray | None = None, delta_score: ndarray | None = None, target_labels: ndarray | None = None, labels: ndarray | None = None, ** kwargs):
        if delta_score is None:
            delta_score = cls.delta_score(field)
        if normalized_peak_value is None:
            normalized_peak_value = cls.difficulty_peaks(
                field, target_labels=route_labels, labels=labels, **kwargs)[-1]
        stable = (Constant.min_max_normalize(delta_score) + 1) * 0.5
        sources = np.zeros_like(field)

        def d_norm(selector):
            max_peak = (normalized_peak_value > 0)
            stable[selector] += normalized_peak_value[selector]*0.1
            sources[selector & max_peak] = normalized_peak_value[selector &
                                                                 max_peak]*0.5*source_amount

        if target_labels is None or labels is None:
            d_norm(field == 0)
        else:
            cls.apply_each_areas(target_labels, labels,
                                 lambda selector, lbl: d_norm(selector))
        
        return cls.fluid(sources, stable, field == 0, steps=steps, fmax=source_amount, fmin=-source_amount)
    
    @classmethod
    def set_start_goal(cls, field: ndarray, filled_steps: ndarray, normalized_peak_value: ndarray, target_labels: ndarray | None = None, labels: ndarray | None = None):
        result = np.zeros_like(field)
        points = {}
        
        def set_sg(selector, lbl):
            max_point = (filled_steps == np.min(
                filled_steps[selector])) & selector
            max_point_values = (normalized_peak_value == np.max(
                normalized_peak_value[max_point])) & max_point
            min_point = (filled_steps == np.max(
                filled_steps[selector])) & selector
            min_point_values = min_point & (normalized_peak_value == np.min(normalized_peak_value[min_point]))
            result[max_point_values] = 1
            result[min_point_values] = -1
            points[lbl] = [np.argwhere(min_point_values), np.argwhere(max_point_values)]
        
        if target_labels is None or labels is None:
            set_sg(field == 0, 0)
        else:
            cls.apply_each_areas(target_labels, labels,
                                 lambda selector, lbl: set_sg(selector, lbl))
        
        return result, points
    
    @classmethod
    def create_maze(cls, shape:tuple[int,int], min_size:int):
        field = np.zeros(shape)
        res = Constant().auto_setting(field, min_size, 300)
        result, route_labels, labels = res[-1]

        # Generate start and goal positions
        d_score, _ = Analyzer.difficulty(
            result, steps=5000, target_labels=route_labels, labels=labels)
        _, _, normalized_peak_value = Analyzer.difficulty_peaks(
            result, d_score, target_labels=route_labels, labels=labels)
        _, _, fluid_label = Analyzer.fluid_difficulty(result, steps=5000, normalized_peak_value=normalized_peak_value,
                                                      difficulty_score=d_score, target_labels=route_labels, labels=labels)
        sg_result, sg_points = Analyzer.set_start_goal(field, fluid_label, normalized_peak_value,
                                                       target_labels=route_labels, labels=labels)
        return (result, labels, sg_points)
            

class Enemy:
    def __init__(self, pos: tuple[int, int], move_type: Literal['Random', 'TurnAlternation', 'LHandApproach', 'RHandApproach', 'StraightOccasionalRandom', None] = None) -> None:
        self.pos = pos
        self.direc_table = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.direc = random.choice(self.direc_table)
        self.move_type = move_type if move_type is not None else random.choice(
            ['Random', 'TurnAlternation', 'LHandApproach', 'RHandApproach', 'StraightOccasionalRandom'])
        self.speed = 1
        self.stock = 0.0
        self.moved = False
        self.v = self.speed/MazeGame.FPS

    def next(self, maze: ndarray):
        self.stock += self.v
        self.moved = False
        if self.stock >= 1:
            self.stock = 0.0
            npi, npj = (pij+dij for pij, dij in zip(self.pos, self.direc))
            if self.is_valid_pos(maze, (npi, npj)):
                self.pos = (npi, npj)
            self.choice_direc(maze)
            self.moved = True

    def get_game_pos(self):
        return (self.pos[1],self.pos[0])

    def is_valid_pos(self, maze: ndarray, pos: tuple[int, int]) -> bool:
        i, j = pos
        return 0 <= i < maze.shape[0] and 0 <= j < maze.shape[1] and maze[i, j] == 0

    def choice_direc(self, maze: ndarray):
        if self.move_type == 'Random':
            self.direc = random.choice(self.direc_table)

        elif self.move_type == 'TurnAlternation':
            npi, npj = (pij + dij for pij, dij in zip(self.pos, self.direc))
            if self.is_valid_pos(maze, (npi, npj)):
                return
            target_range = list(range(len(self.direc_table)-1))
            random.shuffle(target_range)
            for n in target_range:
                self.turn_alternation_state = (
                    self.direc_table.index(self.direc) + n + 1) % len(self.direc_table)
                if self.is_valid_pos(maze, (npi, npj)):
                    self.direc = self.direc_table[self.turn_alternation_state]
                    return
            self.direc = random.choice(self.direc_table)
        elif self.move_type == 'LHandApproach' or self.move_type == 'RHandApproach':
            left_turn = {(0, 1): (-1, 0), (1, 0): (0, 1),
                         (0, -1): (1, 0), (-1, 0): (0, -1)}
            right_turn = {(-1, 0): (0, 1), (0, -1): (-1, 0),
                          (1, 0): (0, -1), (0, 1): (1, 0)}

            main_turn = left_turn if self.move_type == 'LHandApproach' else right_turn
            sub_turn = right_turn if self.move_type == 'LHandApproach' else left_turn
            main_direc = main_turn[self.direc]
            npi, npj = (pij + dij for pij, dij in zip(self.pos, main_direc))
            if self.is_valid_pos(maze, (npi, npj)):
                self.direc = main_direc
                if random.random() < 0.2:
                    self.direc = random.choice(self.direc_table)
                return

            npi, npj = (pij + dij for pij, dij in zip(self.pos, self.direc))
            if self.is_valid_pos(maze, (npi, npj)):
                return

            self.direc = sub_turn[self.direc]
        elif self.move_type == 'StraightOccasionalRandom':
            if random.random() < 0.5:
                self.direc = random.choice(self.direc_table)
            else:
                npi, npj = (pij + dij for pij,
                            dij in zip(self.pos, self.direc))
                if not self.is_valid_pos(maze, (npi, npj)):
                    self.direc = random.choice(self.direc_table)


class GameItem:
    def __init__(self, name: str, cooldown: int, duration: int, sound_name:str|None=None):
        self.name = name
        self.cooldown = cooldown * MazeGame.FPS
        self.duration = duration * MazeGame.FPS
        self.current_cooldown = 0
        self.current_time = 0
        self.sound = None if sound_name is None else (
            pygame.mixer.Sound(MazeGame.SOUND_DIR+sound_name))

    def use(self, no_draw: bool = False):
        if self.current_cooldown == 0 and self.current_time == 0:
            self.current_cooldown = self.cooldown
            self.current_time = self.duration
            if self.sound and not no_draw:
                self.sound.play()
            return True
        return False

    def update(self):
        if self.current_time > 0:
            self.current_time -= 1
        elif self.current_cooldown > 0:
            self.current_cooldown -= 1
    
    def get_busy(self):
        return self.current_time > 0


class MonsterVisionItem(GameItem):
    def __init__(self):
        super().__init__("Monster Vision", 10, 5, "vision_monster.mp3")  # 10 seconds cooldown


class ExtraLightItem(GameItem):
    def __init__(self):
        super().__init__("Extra Light", 10, 20, "extra_light.mp3")  # 20 seconds cooldown
        self.extra_light = 10
        self.current_extra_light = 0.0

    def use(self, no_draw: bool = False):
        usable = super().use(no_draw)
        if usable:
            self.current_extra_light = self.extra_light
        return usable
    
    def update(self):
        if super().get_busy():
            self.current_extra_light = self.extra_light * ((self.current_time - 1)/self.duration)
        else:
            self.current_extra_light = 0.0
        super().update()

class PathfinderItem(GameItem):
    def __init__(self):
        super().__init__("Pathfinder", 30, 5, "pathfinder.mp3")  # 30 seconds cooldown

class PlayerStatus:
    def __init__(self, max_mp, max_sight, initial_sight, max_item:int=2):
        self.pos = (0, 0)
        self.max_mp = max_mp
        self.mp = max_mp
        self.max_sight = max_sight
        self.sight = initial_sight
        self.extra_sight = 0
        self.teleport_mode = False
        self.transparent_timer = 0
        self.items:list[GameItem] = [None for _ in range(max_item)]

    def move(self, new_pos):
        self.pos = new_pos

    def use_mp(self, amount):
        self.mp = max(0, self.mp - amount)

    def restore_mp(self, amount):
        self.mp = min(self.max_mp, self.mp + amount)

    def reduce_sight(self, amount):
        self.sight = max(0, self.sight - amount)

    def restore_sight(self, amount):
        self.sight = min(self.max_sight, self.sight + amount)

    def set_extra_sight(self, amount):
        self.extra_sight = amount

    def add_item(self, item):
        for i in range(len(self.items)):
            if self.items[i] is None:
                self.items[i] = item
                return True
        return False

    def use_item(self, slot: int, no_draw: bool = False):
        if 0 <= slot < len(self.items) and self.items[slot] is not None:
            if self.items[slot].use(no_draw):
                return True
        return False

    def update_items(self):
        for item in self.items:
            if item is not None:
                item.update()

    def get_total_sight(self):
        return self.sight + self.extra_sight + sum([(item.current_extra_light if hasattr(item, 'current_extra_light') else 0) for item in self.items])
    
    def get_busy_item(self, item_name: str):
        for item in self.items:
            if item.name == item_name and item.get_busy():
                return True
        return False

    def get_vision_monster(self):
        return self.get_busy_item("Monster Vision")
    
    def get_vision_path(self):
        return self.get_busy_item("Pathfinder")

    def set_teleport_mode(self, mode):
        self.teleport_mode = mode

    def set_transparent_timer(self, time):
        self.transparent_timer = time

    def update_transparent_timer(self):
        if self.transparent_timer > 0:
            self.transparent_timer -= 1

    def is_transparent(self):
        return self.transparent_timer > 0

class MazeGame:

    # 色の定義
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    CELL_SIZE = 50
    ITEM_BOX_SIZE = 90
    ITEM_BOX_MARGIN = 10
    FPS = 30
    MAX_SIZE = 800
    SOUND_DIR = "sounds/"
    # フォントの設定
    FONT_DIR = "C:/Windows/Fonts"
    # HGゴシック
    FONT_NAME = "HGRGM.TTC"
    @staticmethod
    def FONT_PATH(): return os.path.join(MazeGame.FONT_DIR, MazeGame.FONT_NAME)
    FONTSIZE = 24

    def __init__(self, maze: ndarray, regions: ndarray, start_goal_candidates:dict[int,list[ndarray,ndarray]], sight:int=2) -> None:
        MazeGame.CELL_SIZE = min(MazeGame.CELL_SIZE,MazeGame.MAX_SIZE//max(maze.shape))
        self.SCREEN_WIDTH = maze.shape[1] * MazeGame.CELL_SIZE
        self.SCREEN_HEIGHT = maze.shape[0] * MazeGame.CELL_SIZE
        self.maze = maze.copy()
        self.max_sight = sight
        self.regions = regions
        self.start_goal_candidates = start_goal_candidates
        self.reset()
        
    def reset(self, no_draw:bool=False):
        # ヒント表示用の変数
        self.hint_timer = 0
        self.hint_duration = 3 * MazeGame.FPS  # 3秒間（30FPSと仮定）
        self.hint_mp = 10
        self.restore_mpf = 2 / MazeGame.FPS
        self.max_mp = 20
        self.transparent_timer = 0
        self.max_transparent_time = 1 * MazeGame.FPS
        self.monster_adding_time = 0
        self.monster_adding_interval = 30 * MazeGame.FPS 
        # MPゲージの設定
        self.mp_gauge_height = 20
        self.mp_gauge_margin = 10
        self.teleport_mp_cost = 5
        self.teleport_sight_cost = 0.5
        self.min_mp_for_teleport = 10
        self.min_sight_for_teleport = 1.5
        self.sight_recovery_rate = 0.4 / MazeGame.FPS  # 1フレームあたりの回復量
        self.light_dispersion = 0.6  # 光の分散の強さ (0.0 ~ 1.0)
        self.extra_sight = 0
        self.mp_to_brightness_rate = 1 / MazeGame.FPS   # 1 MPあたりの追加視界
        self.mp_to_brightness_cost = 5 / MazeGame.FPS  # 1フレームあたりのMP消費量
        self.min_mp_for_brightness = 5  # この能力を使うために必要な最小MP
        self.enemy_count = 3
        self.enemy_damage = self.max_sight/2
        self.enemies: list[Enemy] = []
        self.player = PlayerStatus(self.max_mp, self.max_sight, self.max_sight)
        self.action_log = []
        self.enemy_log = []
        import datetime
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        self.action_log_dir = "log/"
        if not os.path.exists(self.action_log_dir):
            os.makedirs(self.action_log_dir)
        date_str = now.strftime('%Y%m%d%H%M%S')
        self.action_log_name = os.path.join(
            self.action_log_dir, "player_actions{}.csv".format(date_str))
        self.action_log_field_name = os.path.join(
            self.action_log_dir, "player_actions_field{}.csv".format(date_str))
        self.start_time = None
        self.game_init(no_draw)
    
    def initialize_items(self):
        items = [MonsterVisionItem(), ExtraLightItem(), PathfinderItem()]
        random.shuffle(items)
        self.player.add_item(items[0])
        self.player.add_item(items[1])
    
    def game_init(self,no_draw:bool=False):
        # Pygameの初期化
        pygame.init()
        if not no_draw:
            self.screen = pygame.display.set_mode(
                (self.SCREEN_WIDTH + MazeGame.ITEM_BOX_MARGIN * 2 + MazeGame.ITEM_BOX_SIZE, self.SCREEN_HEIGHT + (self.mp_gauge_height + self.mp_gauge_margin)*2))
            pygame.display.set_caption("Maze Game")
            # Pygameのミキサーを初期化
            pygame.mixer.init()
            # 効果音の読み込み
            self.sounds = {
                'hint': pygame.mixer.Sound(f'{MazeGame.SOUND_DIR}hint.mp3'),
                'goal': pygame.mixer.Sound(f'{MazeGame.SOUND_DIR}goal.mp3'),
                'teleport': pygame.mixer.Sound(f'{MazeGame.SOUND_DIR}teleport.mp3'),
                'game_over': pygame.mixer.Sound(f'{MazeGame.SOUND_DIR}game_over.mp3'),
                'hit_enemy': pygame.mixer.Sound(f'{MazeGame.SOUND_DIR}hit_enemy.mp3'),
                'light': pygame.mixer.Sound(f'{MazeGame.SOUND_DIR}light.mp3'),
                'light_end': pygame.mixer.Sound(f'{MazeGame.SOUND_DIR}light_end.mp3'),
                'monster_move': pygame.mixer.Sound(f'{MazeGame.SOUND_DIR}monster_move.mp3')
            }
            MazeGame.FONT = pygame.font.Font(MazeGame.FONT_PATH(), MazeGame.FONTSIZE)
        self.initialize_items()

    def initialize_enemies(self, mask):
        # 敵の初期位置をランダムに選択（壁でない場所）
        poses = np.argwhere(mask)
        while len(self.enemies) < self.enemy_count:
            self.enemies.append(Enemy(random.choice(poses)))
    
    def initialize_enemy(self):
        poses = np.argwhere(self.regions == self.region)
        self.enemies.append(Enemy(random.choice(poses)))

    def log_action(self, action_type: str, details: dict = None):
        if self.start_time is None:
            self.start_time = pygame.time.get_ticks()

        timestamp = (pygame.time.get_ticks() - self.start_time) / 1000  # 秒単位
        log_entry = {
            "timestamp": timestamp,
            "action": action_type,
            "player_pos": self.player.pos,
            "player_sight": self.player.sight,
            "player_mp": self.player.mp
        }
        if details:
            log_entry.update(details)
        self.action_log.append(log_entry)
        check_maze = np.zeros_like(self.maze,dtype=int)
        for enemy in self.enemies:
            check_maze[enemy.pos] = 1
        check_maze_str = f"{check_maze.tolist()}"
        self.enemy_log.append(f"{timestamp},{check_maze_str}")

    def save_action_log(self):
        import csv
        print(self.action_log)
        with open(self.action_log_name, 'w', newline='') as csvfile:
            fieldnames = ["timestamp", "action",
                          "player_pos", "player_sight", "player_mp"]
            for entry in self.action_log:
                for key in entry.keys():
                    if not key in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for entry in self.action_log:
                writer.writerow(entry)
        with open(self.action_log_field_name, 'w') as csvfile:
            csvfile.write(f"{self.maze.shape[0]},{self.maze.shape[1]}")
            csvfile.write(f"{self.maze.astype(int).tolist()}")
            csvfile.writelines(self.enemy_log)

    def move(self, new_pos):
        old_pos = self.player.pos
        self.player.move(new_pos)
        self.log_action("move", {"from": old_pos, "to": new_pos})
    
    def use_item(self, slot: int, no_draw: bool=False):
        if self.player.use_item(slot, no_draw):
            self.log_action("use_item", {"slot": slot, "item": self.player.items[slot].name})
            return True
        return False
    
    def draw_enemy(self, visibility):
        all_look = self.player.get_vision_monster()
        for enemy in self.enemies:
            is_visible = self.is_visible_from_player(enemy.get_game_pos(), visibility)
            if is_visible or all_look:
                enemy_rect = pygame.Rect(
                    enemy.pos[1] *
                    MazeGame.CELL_SIZE, enemy.pos[0] * MazeGame.CELL_SIZE,
                    MazeGame.CELL_SIZE, MazeGame.CELL_SIZE)
                pygame.draw.rect(self.screen, MazeGame.BLUE, enemy_rect)
                if enemy.moved and is_visible:
                    self.play_sound('monster_move', volume=visibility[*enemy.pos])

    def update_enemy(self):
        for enemy in self.enemies:
            enemy.next(self.maze)

    def check_collision(self, pos: tuple[int,int]|None=None):
        if pos is None:
            pos = self.player.pos
        for enemy in self.enemies:
            if pos == enemy.get_game_pos():
                return True
        return False

    def teleport(self, direction, no_draw:bool = False):
        old_pos = self.player.pos
        x, y = self.player.pos
        dx, dy = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0)
        }[direction]

        max_distance = int(np.ceil(self.player.sight / self.teleport_sight_cost)) - 1
        if max_distance <= 1:
            return
        
        def move_to(distance):
            self.player.pos = (x + dx * distance,
                               y + dy * distance)
            self.player.sight -= distance * self.teleport_sight_cost
            self.player.mp -= self.teleport_mp_cost
            if not no_draw:
                self.play_sound('teleport')

        for distance in range(1, max_distance + 1):
            new_x, new_y = x + dx * distance, y + dy * distance
            if (new_x < 0 or new_x >= self.maze.shape[1] or
                new_y < 0 or new_y >= self.maze.shape[0] or
                    self.maze[new_y, new_x] == 1):
                # 壁にぶつかるか迷路の外に出る直前で停止
                distance = distance - 1
                break
        to_pos = (x + dx * distance, y + dy * distance)
        while distance > 0 and self.check_collision(to_pos):
            distance -= 1
            to_pos = (x + dx * distance, y + dy * distance)
        if distance > 0 and not self.check_collision(to_pos):
            move_to(distance)
        self.log_action(
            "teleport", {"from": old_pos, "to": self.player.pos, "direction": direction})
        
    def handle_teleport(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q] and self.player.mp >= self.min_mp_for_teleport and self.player.sight >= self.min_sight_for_teleport and self.player.extra_sight == 0:
            self.player.set_teleport_mode(True)
            self.draw()
            pygame.display.flip()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            self.teleport('UP')
                            waiting = False
                        elif event.key == pygame.K_DOWN:
                            self.teleport('DOWN')
                            waiting = False
                        elif event.key == pygame.K_LEFT:
                            self.teleport('LEFT')
                            waiting = False
                        elif event.key == pygame.K_RIGHT:
                            self.teleport('RIGHT')
                            waiting = False
                        elif event.key == pygame.K_q:
                            waiting = False
            q_release_wait = pygame.key.get_pressed()[pygame.K_q]
            while q_release_wait:
                for event in pygame.event.get():
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_q:
                            q_release_wait = False
            self.player.set_teleport_mode(False)
    
    def handle_teleport_for_ai(self, select: int, no_draw:bool=True):
        if self.player.mp >= self.min_mp_for_teleport and self.player.sight >= self.min_sight_for_teleport and self.player.extra_sight == 0:
            self.player.set_teleport_mode(True)
            if not no_draw:
                self.draw()
                pygame.display.flip()
            if select == 0:
                self.teleport('UP', no_draw)
            elif select == 1:
                self.teleport('DOWN',no_draw)
            elif select == 2:
                self.teleport('LEFT', no_draw)
            elif select == 3:
                self.teleport('RIGHT',no_draw)
            self.player.set_teleport_mode(False)

    def handle_mp_to_brightness(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_e] and self.player.mp >= self.min_mp_for_brightness and not self.player.teleport_mode:
            if self.player.extra_sight == 0:
                self.play_sound('light')
            self.player.use_mp(self.mp_to_brightness_cost)
            extra_sight = (
                self.player.mp - self.min_mp_for_brightness) * self.mp_to_brightness_rate
            self.player.set_extra_sight(extra_sight)
        else:
            if self.player.extra_sight > 0:
                self.play_sound('light_end')
            self.player.set_extra_sight(0)
    
    def handle_mp_to_brightness_for_ai(self, selected: bool, no_draw:bool=True):
        if selected and self.player.mp >= self.min_mp_for_brightness and not self.player.teleport_mode:
            if self.player.extra_sight == 0 and not no_draw:
                self.play_sound('light')
            self.player.use_mp(self.mp_to_brightness_cost)
            extra_sight = (
                self.player.mp - self.min_mp_for_brightness) * self.mp_to_brightness_rate
            self.player.set_extra_sight(extra_sight)
        else:
            if self.player.extra_sight > 0 and not no_draw:
                self.play_sound('light_end')
            self.player.set_extra_sight(0)
    
    def handle_hint(self, no_draw:bool=False):
        if self.hint_timer <= 0 and self.player.mp >= self.hint_mp:
            self.hint_timer = self.hint_duration
            self.player.use_mp(self.hint_mp)
            if not no_draw:
                self.play_sound('hint')
                self.log_action("use_hint")

    def draw(self):
        self.screen.fill(MazeGame.BLACK)
        total_sight = self.player.get_total_sight()
        visibility = self.simulate_light_propagation(
            self.maze, (self.player.pos[1], self.player.pos[0]), total_sight)
        self.draw_maze_with_visibility(visibility)
        self.draw_player(*self.player.pos)

        if self.is_visible_from_player(self.goal_pos, visibility):
            goal_rect = pygame.Rect(
                self.goal_pos[0] * MazeGame.CELL_SIZE, self.goal_pos[1] * MazeGame.CELL_SIZE, MazeGame.CELL_SIZE, MazeGame.CELL_SIZE)
            pygame.draw.rect(self.screen, MazeGame.GREEN, goal_rect)

        if self.hint_timer > 0:
            self.draw_hint_arrow(self.player.pos, self.goal_pos)

        self.draw_mp_gauge(self.player.mp)
        self.draw_sight_gauge(self.player.sight)
        self.draw_enemy(visibility)

        if self.player.teleport_mode:
            self.draw_teleport_options()

        if self.player.get_vision_path():
            self.draw_path_to_goal()

        self.draw_items()
    
    def draw_items(self):
        for i, item in enumerate(self.player.items):
            if item is not None:
                item_rect = pygame.Rect(
                    self.SCREEN_WIDTH + MazeGame.ITEM_BOX_MARGIN,
                    MazeGame.ITEM_BOX_MARGIN + (MazeGame.ITEM_BOX_SIZE + MazeGame.ITEM_BOX_MARGIN)*i,
                    MazeGame.ITEM_BOX_SIZE, MazeGame.ITEM_BOX_SIZE
                )
                pygame.draw.rect(self.screen, MazeGame.WHITE, item_rect, 2)
                item_text = MazeGame.FONT.render(
                    f"{i+1}:{item.name}", True, MazeGame.WHITE)
                self.screen.blit(item_text, item_rect)

                cooldown_rect = pygame.Rect(
                    item_rect.left, item_rect.bottom+5,
                    MazeGame.ITEM_BOX_SIZE * (1 - item.current_cooldown / item.cooldown), 5
                )
                pygame.draw.rect(self.screen, MazeGame.GREEN, cooldown_rect)
    
    def draw_path_to_goal(self):
        path = self.shortest_path(self.player.pos, self.goal_pos, max_depth=int(self.player.sight)*2)
        if path:
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                start_pos = (start[0] * MazeGame.CELL_SIZE + MazeGame.CELL_SIZE // 2,
                             start[1] * MazeGame.CELL_SIZE + MazeGame.CELL_SIZE // 2)
                end_pos = (end[0] * MazeGame.CELL_SIZE + MazeGame.CELL_SIZE // 2,
                           end[1] * MazeGame.CELL_SIZE + MazeGame.CELL_SIZE // 2)
                pygame.draw.line(self.screen, MazeGame.YELLOW,
                                 start_pos, end_pos, 2)
                
    def shortest_path(self, start, goal, max_depth):
        rows, cols = self.maze.shape
        start = (start[1], start[0])  # Convert to (y, x) format
        goal = (goal[1], goal[0])  # Convert to (y, x) format
        queue = deque([(start, [(start[1], start[0])])])
        visited = set([start])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上

        while queue:
            (i, j), path = queue.popleft()
            if (i, j) == goal:
                if len(path) < max_depth:
                    return path
                return path[:max_depth]

            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and self.maze[ni,nj] == 0 and (ni, nj) not in visited:
                    queue.append(((ni, nj), path + [(nj, ni)]))
                    visited.add((ni, nj))

        return None  # 経路が見つからない場合

    def draw_sight_gauge(self, player_sight):
        gauge_width = self.SCREEN_WIDTH - 2 * self.mp_gauge_margin
        gauge_rect = pygame.Rect(self.mp_gauge_margin, self.SCREEN_HEIGHT + 2 * self.mp_gauge_margin + self.mp_gauge_height,
                                 gauge_width, self.mp_gauge_height)

        pygame.draw.rect(self.screen, MazeGame.BLUE, gauge_rect)

        current_sight_width = int(
            gauge_width * (player_sight / self.max_sight))
        current_sight_rect = pygame.Rect(self.mp_gauge_margin, self.SCREEN_HEIGHT + 2 * self.mp_gauge_margin + self.mp_gauge_height,
                                         current_sight_width, self.mp_gauge_height)
        pygame.draw.rect(self.screen, MazeGame.YELLOW, current_sight_rect)

        sight_text = MazeGame.FONT.render(
            f"Sight: {player_sight:.2f}/{self.max_sight}", True, MazeGame.WHITE)
        text_rect = sight_text.get_rect(center=(self.SCREEN_WIDTH // 2,
                                                self.SCREEN_HEIGHT + 2 * self.mp_gauge_margin + self.mp_gauge_height * 1.5))
        self.screen.blit(sight_text, text_rect)

    def draw_teleport_options(self):
        text = MazeGame.FONT.render(
            "Choose teleport direction (↑↓←→)", True, MazeGame.WHITE)
        text_rect = text.get_rect(
            center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        self.screen.blit(text, text_rect)

    def play_sound(self, sound_name, wait:bool=False, volume:float=1.0):
        if sound_name in self.sounds:
            sound = self.sounds[sound_name]
            sound.set_volume(volume)
            ch = sound.play()
            if wait:
                while ch.get_busy():
                    self.clock.tick(MazeGame.FPS)

    
    def draw_mp_gauge(self, player_mp):
        gauge_width = self.SCREEN_WIDTH - 2 * self.mp_gauge_margin
        gauge_rect = pygame.Rect(self.mp_gauge_margin, self.SCREEN_HEIGHT + self.mp_gauge_margin,
                                 gauge_width, self.mp_gauge_height)

        # 背景（最大MP）を描画
        pygame.draw.rect(self.screen, MazeGame.BLUE, gauge_rect)

        # 現在のMPを描画
        current_mp_width = int(gauge_width * (player_mp / self.max_mp))
        current_mp_rect = pygame.Rect(self.mp_gauge_margin, self.SCREEN_HEIGHT + self.mp_gauge_margin,
                                      current_mp_width, self.mp_gauge_height)
        pygame.draw.rect(self.screen, MazeGame.GREEN, current_mp_rect)

        # MPの数値を表示
        mp_text = MazeGame.FONT.render(
            f"MP: {int(player_mp)}/{self.max_mp}", True, MazeGame.WHITE)
        text_rect = mp_text.get_rect(center=(self.SCREEN_WIDTH // 2,
                                             self.SCREEN_HEIGHT + self.mp_gauge_margin + self.mp_gauge_height // 2))
        self.screen.blit(mp_text, text_rect)

    
    def gaussian_like_brightness(self,distance, max_distance):
        if distance > max_distance:
            return 0
        return np.exp(-(distance ** 2) / (2 * (max_distance / 2) ** 2))
    
    def is_visible(self, x1, y1, x2, y2):
        """ブレゼンハムのライン アルゴリズムを使用して、2点間の視線が遮られていないかチェック"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        while n > 1:
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

            if self.maze[y, x] == 1:  # 壁に当たったら視線が遮られる
                return False
            n -= 1

        return True
    
    def is_visible_from_player(self, target_pos: tuple[int, int], visibility):
        if visibility[target_pos[1],target_pos[0]] > 0.05:
                return True
        return False
    
    def simulate_light_propagation(self, maze, light_source, intensity: float):
        light = np.zeros_like(maze, dtype=float)
        mask = maze == 0
        light[light_source] = max(0,intensity)

        kernel = np.array([
            [0, 0.5, 0],
            [0.5, 1, 0.5],
            [0, 0.5, 0]])
        kernel /= np.sum(kernel)

        num_iterations = int(np.ceil(np.sqrt(abs(intensity))))

        for _ in range(num_iterations):
            new_light = signal.convolve2d(light * mask, kernel, mode='same')
            new_light[light_source] = intensity
            new_light[~mask] = 0
            light = new_light

        return np.clip(light, 0, 1)

    def draw_maze_with_visibility(self, visibility):
        for y in range(self.maze.shape[0]):
            for x in range(self.maze.shape[1]):
                rect = pygame.Rect(x * MazeGame.CELL_SIZE, y * MazeGame.CELL_SIZE,
                                   MazeGame.CELL_SIZE, MazeGame.CELL_SIZE)

                if self.maze[y, x] == 1:  # 壁
                    color = tuple(
                        int(255 * visibility[y, x]) for _ in range(3))
                    pygame.draw.rect(self.screen, color, rect)
                else:  # 通路
                    color = tuple(
                        int(255 * visibility[y, x]) for _ in range(3))
                    pygame.draw.rect(self.screen, color, rect)

    def draw_player(self, x, y):
        rect = pygame.Rect(x * MazeGame.CELL_SIZE, y *
                           MazeGame.CELL_SIZE, MazeGame.CELL_SIZE, MazeGame.CELL_SIZE)
        if self.player.extra_sight > 0:
            pygame.draw.rect(self.screen, MazeGame.YELLOW if self.player.transparent_timer %
                             5 == 0 else MazeGame.MAGENTA, rect)
        else:
            pygame.draw.rect(self.screen, MazeGame.RED if self.player.transparent_timer %
                             5 == 0 else MazeGame.MAGENTA, rect)

    def draw_hint_arrow(self, player_pos, goal_pos):
        start_pos, end_pos, dx,dy = self.get_hint_arrow()

        # 矢印を描画
        pygame.draw.line(self.screen, MazeGame.YELLOW, start_pos, end_pos, 3)

        # 矢印の先端を描画
        if end_pos[0] != start_pos[0]:
            tip_y = start_pos[1]
            tip_x = end_pos[0] + 10 * (-1 if dx > 0 else 1)
            pygame.draw.line(self.screen, MazeGame.YELLOW,
                             end_pos, (tip_x, tip_y - 10), 3)
            pygame.draw.line(self.screen, MazeGame.YELLOW,
                             end_pos, (tip_x, tip_y + 10), 3)
        else:
            tip_x = start_pos[0]
            tip_y = end_pos[1] + 10 * (-1 if dy > 0 else 1)
            pygame.draw.line(self.screen, MazeGame.YELLOW,
                             end_pos, (tip_x - 10, tip_y), 3)
            pygame.draw.line(self.screen, MazeGame.YELLOW,
                             end_pos, (tip_x + 10, tip_y), 3)
    
    def get_hint_arrow(self):
        px, py = self.player.pos
        gx, gy = self.goal_pos

        # プレイヤーとゴールの相対位置を計算
        dx = gx - px
        dy = gy - py

        # 矢印の始点（プレイヤーの位置）
        start_pos = (px * MazeGame.CELL_SIZE + MazeGame.CELL_SIZE // 2,
                     py * MazeGame.CELL_SIZE + MazeGame.CELL_SIZE // 2)

        # 矢印の終点（ヒントとしての方向）
        length = min(max(abs(dx), abs(dy)), 3) * MazeGame.CELL_SIZE
        if abs(dx) > abs(dy):
            end_pos = (start_pos[0] + length *
                       (1 if dx > 0 else -1), start_pos[1])
        else:
            end_pos = (start_pos[0], start_pos[1] +
                       length * (1 if dy > 0 else -1))
        return start_pos, end_pos, dx, dy

            
    def setup(self):
        # プレイヤーの初期位置をランダムに選択
        self.region = random.choice(list(self.start_goal_candidates.keys()))
        player_pos = random.choice(self.start_goal_candidates[self.region][0])
        goal_pos = random.choice(self.start_goal_candidates[self.region][1])

        self.player.pos = (player_pos[1].item(), player_pos[0].item())
        self.goal_pos = (goal_pos[1].item(), goal_pos[0].item())

        appear_mask = self.regions == self.region
        appear_mask[player_pos] = False
        self.enemy_count = np.sum(self.regions == self.region) // 8
        self.initialize_enemies(appear_mask)
        self.clock = pygame.time.Clock()

    def main(self):
        self.setup()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_1, pygame.K_2] and not self.player.teleport_mode:
                        slot = event.key - pygame.K_1
                        self.use_item(slot)
                    elif not self.player.teleport_mode and self.player.extra_sight == 0:
                        x, y = self.player.pos
                        if event.key == pygame.K_UP and y > 0 and self.maze[y-1, x] == 0:
                            self.move((x, y-1))
                        elif event.key == pygame.K_DOWN and y < self.maze.shape[0]-1 and self.maze[y+1, x] == 0:
                            self.move((x, y+1))
                        elif event.key == pygame.K_LEFT and x > 0 and self.maze[y, x-1] == 0:
                            self.move((x-1, y))
                        elif event.key == pygame.K_RIGHT and x < self.maze.shape[1]-1 and self.maze[y, x+1] == 0:
                            self.move((x+1, y))
                        elif event.key == pygame.K_SPACE:
                            self.handle_hint()

            self.handle_mp_to_brightness()

            if not self.player.teleport_mode:
                if self.monster_adding_time < self.monster_adding_interval:
                    self.monster_adding_time += 1
                else:
                    self.monster_adding_time = 0
                    self.initialize_enemy()
                self.update_enemy()
                self.handle_teleport()

                self.player.update_transparent_timer()
                self.player.update_items()

                if not self.player.is_transparent():
                    if self.hint_timer > 0:
                        self.hint_timer -= 1

                    self.player.restore_mp(self.restore_mpf)
                    self.player.restore_sight(self.sight_recovery_rate)

                    if self.check_collision():
                        self.log_action("enemy_collision", {
                                        "enemy_pos": self.player.pos})
                        self.play_sound('hit_enemy')
                        self.player.reduce_sight(self.enemy_damage)
                        self.player.set_transparent_timer(
                            self.max_transparent_time)

            self.draw()
            pygame.display.flip()

            if self.player.pos == self.goal_pos:
                self.log_action("goal_reached")
                print("Goal reached!")
                self.play_sound('goal', True)
                running = False

            if self.player.sight <= 0:
                self.log_action("game_over", {"reason": "out_of_sight"})
                print("Game Over! You were eaten by the monster.")
                self.play_sound('game_over', True)
                running = False

            self.clock.tick(MazeGame.FPS)

        self.save_action_log()  # ゲーム終了時にログを保存
        pygame.quit()
    
    def step(self, action: int, keep_press:bool, no_draw: bool = True):
        bright_action = 9
        if len(self.player.items)+bright_action >= action > bright_action and not self.player.teleport_mode and not keep_press:
            slot = action - bright_action - 1
            self.use_item(slot, no_draw)
        elif not self.player.teleport_mode and self.player.extra_sight == 0 and not keep_press:
            x, y = self.player.pos
            if action == 0 and y > 0 and self.maze[y-1, x] == 0:
                self.move((x, y-1))
            elif action == 1 and y < self.maze.shape[0]-1 and self.maze[y+1, x] == 0:
                self.move((x, y+1))
            elif action == 2 and x > 0 and self.maze[y, x-1] == 0:
                self.move((x-1, y))
            elif action == 3 and x < self.maze.shape[1]-1 and self.maze[y, x+1] == 0:
                self.move((x+1, y))
            elif action == 4:
                self.handle_hint(no_draw)

        self.handle_mp_to_brightness_for_ai(action == bright_action, no_draw)

        if not self.player.teleport_mode:
            self.update_enemy()
            if 4 < action <= 8 and not keep_press:
                self.handle_teleport_for_ai(action-5, no_draw)

            self.player.update_transparent_timer()
            self.player.update_items()

            if not self.player.is_transparent():
                if self.monster_adding_time < self.monster_adding_interval:
                    self.monster_adding_time += 1
                else:
                    self.monster_adding_time = 0
                    self.initialize_enemy()
                if self.hint_timer > 0:
                    self.hint_timer -= 1

                self.player.restore_mp(self.restore_mpf)
                self.player.restore_sight(self.sight_recovery_rate)

                if self.check_collision():
                    if not no_draw:
                        self.play_sound('hit_enemy')
                    self.player.reduce_sight(self.enemy_damage)
                    self.player.set_transparent_timer(
                        self.max_transparent_time)
                
        if not no_draw:
            self.draw()
            pygame.display.flip()

        if self.player.pos == self.goal_pos:
            print("Goal reached!")
            if not no_draw:
                self.play_sound('goal', True)
            return True

        if self.player.sight <= 0:
            print("Game Over! You were eaten by the monster.")
            if not no_draw:
                self.play_sound('game_over', True)
            return True
        
        if not no_draw:
            self.clock.tick(MazeGame.FPS)
        return False


if __name__ == '__main__':
    c = Constant()
    field = np.zeros((30,30))
    res = c.auto_setting(field, 50, 300)
    result, route_labels, labels = res[-1]
    imshow_alpha = (1-result.astype(float))
    """
    plt.subplot(1, 3, 1)
    plt.imshow(result, cmap="grey")
    plt.subplot(1, 3, 2)
    plt.imshow(Analyzer.neighbor_count(result))
    plt.subplot(1, 3, 3)
    plt.imshow(Analyzer.delta_score(result))
    plt.show()
    """
    d_score, d_score_hist = Analyzer.difficulty(
        result, steps=5000, target_labels=route_labels, labels=labels)

    #fig = plt.figure()
    #ani = animation.ArtistAnimation(fig, [[
    #    plt.imshow(1-result, cmap="grey", animated=True),
    #    plt.imshow(r, cmap="gist_heat", animated=True, alpha=imshow_alpha)
    #] for r in d_score_hist], interval=1)
    #plt.show()
    d_max_peak, d_min_peak, normalized_peak_value = Analyzer.difficulty_peaks(
        result, d_score, target_labels=route_labels, labels=labels)
    
    plt.imshow(normalized_peak_value,
               cmap="coolwarm", alpha=imshow_alpha)
    plt.show()
    fluid_result, fluid_hist, fluid_label = Analyzer.fluid_difficulty(
        result,steps=5000, normalized_peak_value=normalized_peak_value, difficulty_score=d_score, target_labels=route_labels, labels=labels
    )
    #fig = plt.figure()
    #ani = animation.ArtistAnimation(fig, [[
    #    plt.imshow(1-result, cmap="grey", animated=True),
    #    plt.imshow(r, cmap="winter", animated=True,
    #               alpha=imshow_alpha,vmin=0,vmax=2)
    #] for r in fluid_hist], interval=1)
    #plt.show()
    """
    plt.imshow(1-fluid_label,
               cmap="grey", alpha=imshow_alpha)
    plt.show()
    """
    sg_result, sg_points = Analyzer.set_start_goal(
        field, fluid_label, normalized_peak_value, target_labels=route_labels, labels=labels)
    plt.imshow(sg_result,
               cmap="coolwarm", alpha=imshow_alpha)
    plt.show()
    print(sg_points)
    mg = MazeGame(result,labels,sg_points,5)
    mg.main()


"""
壁が1,通路を0とした迷路を表したndarrayと、その迷路の各セルがどの領域に含まれるかを表したndarrayと、その迷路のスタート、ゴール地点が各領域ごとに、領域番号をキー、[[スタート地点の候補のindexの集合],[ゴール地点の候補のindexの集合]]をvalueとしたdict,視界の範囲D（4近傍距離）を引数に取り、pygameでその迷路を遊ぶコードを書いてください。
"""


