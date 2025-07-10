import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from DungeonMaker import Constant, Analyzer

def visualize_maze_analysis(final_maze, route_labels, labels, difficulty_score, normalized_peak_value, fluid_result, sg_result):
    """
    迷路生成と分析の最終結果をまとめて可視化する関数
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Maze Generation and Difficulty Analysis', fontsize=16)

    # 1. 最終的に生成された迷路
    axes[0, 0].imshow(final_maze, cmap='binary')
    axes[0, 0].set_title('Generated Maze')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # 2. ラベリングされた領域
    # ラベル0（壁）を黒で表示
    unique_labels = np.unique(labels)
    cmap_labels = plt.cm.get_cmap('viridis', len(unique_labels))
    colors = cmap_labels(np.arange(len(unique_labels)))
    wall_label_index = np.where(unique_labels == labels[final_maze == 1][0] if np.any(final_maze==1) else -1)[0]
    if wall_label_index.size > 0:
        colors[wall_label_index[0]] = [0, 0, 0, 1] # Black for walls
    cmap_labels = ListedColormap(colors)

    axes[0, 1].imshow(labels, cmap=cmap_labels)
    axes[0, 1].set_title('Labeled Areas')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # 3. 難易度スコア (熱拡散)
    im = axes[0, 2].imshow(difficulty_score, cmap='coolwarm')
    axes[0, 2].set_title('Difficulty Score (Heat Diffusion)')
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4. 難易度ピーク
    im = axes[1, 0].imshow(normalized_peak_value, cmap='bwr', vmin=-2, vmax=2)
    axes[1, 0].set_title('Difficulty Peaks (Max:Red, Min:Blue)')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 5. 流体シミュレーション結果
    im = axes[1, 1].imshow(fluid_result, cmap='viridis')
    axes[1, 1].set_title('Fluid Simulation Result')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 6. スタート＆ゴール
    axes[1, 2].imshow(final_maze, cmap='binary')
    # スタート（赤）、ゴール（青）で表示
    cmap_sg = ListedColormap(['blue', 'red'])
    axes[1, 2].imshow(np.ma.masked_where(sg_result == 0, sg_result), cmap=cmap_sg, alpha=0.8)
    axes[1, 2].set_title('Start (Red) & Goal (Blue)')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def animate_process(history, title, cmap='binary'):
    """
    過程をアニメーションで表示する汎用関数
    """
    if not history:
        print(f"No history data to animate for '{title}'.")
        return

    fig, ax = plt.subplots()
    ax.set_title(title)
    im = ax.imshow(history[0], cmap=cmap, animated=True)
    
    # カラーバーを追加（ヒートマップの場合）
    if cmap != 'binary':
        fig.colorbar(im, ax=ax)

    def update(frame):
        im.set_array(history[frame])
        # ヒートマップの場合、カラースケールを更新
        if cmap != 'binary':
            im.set_clim(np.min(history[frame]), np.max(history[frame]))
        return [im]

    anim = FuncAnimation(fig, update, frames=len(history), interval=50, blit=True)
    plt.show()


if __name__ == '__main__':
    # --- 1. パラメータ設定 ---
    MAZE_SHAPE = (20, 20)
    MIN_AREA_SIZE = 50
    GENERATION_STEPS = 400
    DIFFUSION_STEPS = 1000
    FLUID_STEPS = 1000

    # --- 2. 迷路の自動生成 ---
    print("Step 1: Generating maze...")
    c = Constant()
    # 初期フィールド（すべて通路）から開始
    initial_field = np.zeros(MAZE_SHAPE, dtype=np.int8)
    # auto_settingメソッドで迷路を生成
    # 戻り値は (field, route_labels, labels) のタプルのリスト
    generation_history = c.auto_setting(initial_field, min_size=MIN_AREA_SIZE, count=GENERATION_STEPS)
    
    # 最終的な迷路の情報を取得
    final_maze, route_labels, labels = generation_history[-1]
    print("Maze generation complete.")

    # --- 3. 難易度の計算と分析 ---
    # Analyzerクラスのメソッドをdocstringを参考に呼び出す
    print("\nStep 2: Calculating difficulty (Heat Diffusion)...")
    difficulty_score, difficulty_history = Analyzer.difficulty(
        final_maze, steps=DIFFUSION_STEPS, target_labels=route_labels, labels=labels
    )
    
    print("\nStep 3: Identifying difficulty peaks...")
    max_peaks, min_peaks, normalized_peak_value = Analyzer.difficulty_peaks(
        final_maze, difficulty_score=difficulty_score, target_labels=route_labels, labels=labels
    )

    print("\nStep 4: Calculating difficulty (Fluid Simulation)...")
    fluid_result, fluid_history, fluid_label = Analyzer.fluid_difficulty(
        final_maze, 
        steps=FLUID_STEPS, 
        normalized_peak_value=normalized_peak_value, 
        target_labels=route_labels, 
        labels=labels
    )
    
    print("\nStep 5: Setting Start and Goal points...")
    sg_result, sg_points = Analyzer.set_start_goal(
        final_maze, 
        fluid_label, 
        normalized_peak_value,
        target_labels=route_labels, 
        labels=labels
    )
    print("Analysis complete.")

    # --- 4. 結果の可視化 ---
    print("\nDisplaying final results...")
    visualize_maze_analysis(
        final_maze, route_labels, labels, 
        difficulty_score, normalized_peak_value, 
        fluid_result, sg_result
    )

    # --- 5. 過程のアニメーション化 ---
    print("\nDisplaying process animations...")
    # 迷路生成過程
    generation_fields = [hist[0] for hist in generation_history]
    animate_process(generation_fields, title="Maze Generation Process", cmap='binary')

    # 難易度計算過程（熱拡散）
    animate_process(difficulty_history, title="Difficulty Calculation (Heat Diffusion)", cmap='coolwarm')