# PythonDungeonMaker1

*注: この README.md の一部は AI アシスタント (Claude) によって生成されています。*

## 目次

- [PythonDungeonMaker1](#pythondungeonmaker1)
  - [目次](#目次)
  - [概要](#概要)
  - [特徴](#特徴)
  - [アルゴリズムの詳細](#アルゴリズムの詳細)
    - [ダンジョン生成アルゴリズム](#ダンジョン生成アルゴリズム)
    - [難易度解析アルゴリズム](#難易度解析アルゴリズム)
  - [主要なメソッドの説明](#主要なメソッドの説明)
    - [`Analyzer.create_maze(shape: tuple[int, int], min_size: int) -> tuple`](#analyzercreate_mazeshape-tupleint-int-min_size-int---tuple)
    - [`MazeGame.setup(no_draw: bool = False) -> None`](#mazegamesetupno_draw-bool--false---none)
    - [`MazeGame.step(action: int, keep_press: bool, no_draw: bool = True) -> bool`](#mazegamestepaction-int-keep_press-bool-no_draw-bool--true---bool)
  - [ゲームの詳細](#ゲームの詳細)
    - [操作方法](#操作方法)
    - [ゲームのルール](#ゲームのルール)
    - [スコア計算](#スコア計算)
    - [ゲームプレイ戦略](#ゲームプレイ戦略)
    - [カスタマイズ方法](#カスタマイズ方法)
      - [<span style="color: #FF6B6B;">設定のカスタマイズ</span>](#設定のカスタマイズ)
  - [依存ライブラリ](#依存ライブラリ)
  - [プロジェクト構造](#プロジェクト構造)
  - [音声ファイルについて](#音声ファイルについて)
  - [フォント設定](#フォント設定)
    - [Windows ユーザー](#windows-ユーザー)
    - [Mac ユーザー](#mac-ユーザー)
    - [その他のOSユーザー](#その他のosユーザー)
    - [フォント設定に関するトラブルシューティング](#フォント設定に関するトラブルシューティング)
  - [使用方法](#使用方法)
  - [開発者向け情報](#開発者向け情報)
    - [コードの構造](#コードの構造)
    - [カスタマイズと拡張](#カスタマイズと拡張)
  - [ライセンス](#ライセンス)
  - [貢献](#貢献)

## 概要

PythonDungeonMaker1は、プロシージャル生成によるダンジョン（迷路）の作成とそれを利用したゲームを実装したPythonプロジェクトです。このプロジェクトでは、高度なアルゴリズムを用いてランダムなダンジョンを生成し、そのダンジョン内でプレイヤーが探索を行うゲームを提供します。

## 特徴

- プロシージャル生成によるランダムなダンジョン作成
- 難易度解析アルゴリズムによる適切なスタート地点とゴール地点の設定
- プレイヤーの視界、MPなどのリソース管理が必要なゲームプレイ
- 敵キャラクターとの駆け引き
- カスタマイズ可能なゲーム設定

## アルゴリズムの詳細

### ダンジョン生成アルゴリズム

ダンジョン生成アルゴリズムは以下のステップで実装されています：

1. 初期フィールドの作成：全てのセルを通路として初期化
2. ランダムな壁の配置：確率的に壁を配置
3. 領域サイズの検証：各領域が最小サイズを満たすか確認
4. 壁の調整：必要に応じて壁を追加または削除
5. 繰り返し：指定された回数または条件を満たすまで2-4を繰り返す

このアルゴリズムにより、バランスの取れた探索可能な迷路が生成されます。

### 難易度解析アルゴリズム

難易度解析には以下の手法を組み合わせています：

1. 隣接スコア計算：各セルの周囲の開放度を評価
2. デルタスコア計算：隣接スコアの変化率を算出
3. 熱拡散シミュレーション：スコアの全体的な伝播を計算
4. 流体シミュレーション：難易度の可視化と最終的な評価

これらの組み合わせにより、局所的な特徴と全体的な構造の両方を考慮した難易度マップが生成されます。

## 主要なメソッドの説明

### `Analyzer.create_maze(shape: tuple[int, int], min_size: int) -> tuple`

指定されたサイズと最小領域サイズで迷路を生成します。

引数：

- `shape`: 生成する迷路のサイズ（高さ, 幅）
- `min_size`: 最小の領域サイズ

戻り値：

- `tuple`: (迷路のフィールド, ラベル付けされた領域, スタートとゴールの候補位置)

使用例：

```python
maze, labels, start_goal_candidates = Analyzer.create_maze((30, 30), 50)
```

### `MazeGame.setup(no_draw: bool = False) -> None`

ゲームのセットアップを行います。プレイヤーの初期位置、ゴールの位置、敵の配置などを設定します。

引数：

- `no_draw`: 描画を行わない場合はTrue（デフォルト: False）

使用例：

```python
game = MazeGame(maze, labels, start_goal_candidates)
game.setup()
```

### `MazeGame.step(action: int, keep_press: bool, no_draw: bool = True) -> bool`

ゲームを1ステップ進めます。AIによる制御のために使用されます。

引数：

- `action`: 実行するアクション
- `keep_press`: ボタンを押し続けているかどうか
- `no_draw`: 描画を行わない場合はTrue（デフォルト: True）

戻り値：

- `bool`: ゲームが終了した場合はTrue、続行中の場合はFalse

使用例：

```python
game = MazeGame(maze, regions, start_goal_candidates)
game.setup()

# ゲームループ
done = False
while not done:
    # アクションの選択（0: 上, 1: 下, 2: 左, 3: 右, 4: ヒント, 5-8: テレポート, 9: 明るさ増加, 10+: アイテム使用）
    action = ...  # AIまたはユーザー入力によってアクションを決定

    # アクションの実行
    keep_press = False  # 通常は False、長押しの場合は True
    done = game.step(action, keep_press, no_draw=True)

    # ゲーム状態の取得や報酬の計算をここで行う

print("Game finished!")
```

## ゲームの詳細

### 操作方法

- 矢印キー：上下左右に移動
- スペースキー：ヒントを表示（MP消費）
- Qキー：テレポートモードを開始（その後矢印キーで方向を選択）
- Eキー：視界を一時的に広げる（MP消費）
- 1, 2キー：アイテムを使用

### ゲームのルール

1. プレイヤーは迷路内を探索し、ゴールを目指します。
2. プレイヤーには視界とMPがあり、これらを管理しながら進む必要があります。
3. 敵キャラクターに接触すると視界が減少します。視界が0になるとゲームオーバーです。
4. MPを消費してヒントを表示したり、一時的に視界を広げたりできます。
5. テレポートを使用して遠距離移動が可能ですが、MPと視界を消費します。
6. 敵キャラクターは時間経過とともに増加します。
7. ゴールに到達するとゲームクリアです。

### スコア計算

ゲーム終了時に以下の要素を考慮してスコアが計算されます：

- 基本スコア
- 経過時間によるペナルティ
- 残りの視界によるボーナス
- 残りのMPによるボーナス

### ゲームプレイ戦略

1. リソース管理：
   - MPとSightのバランスを取りながら探索を進めることが重要です。
   - 長期的な探索のためにMPを温存するか、即時的な利益のために使用するかの判断が鍵となります。

2. 敵の回避：
   - 敵との接触は視界を大幅に減少させるため、可能な限り回避することが重要です。
   - 敵の移動パターンを学習し、予測することで効率的な回避が可能になります。

3. アイテムの効果的な使用：
   - Monster Vision：敵の位置を把握するのに有効。危険な状況を事前に回避できます。
   - Extra Light：視界が極端に狭くなった際の救済手段として有効です。
   - Pathfinder：ゴールへの最短経路を示すため、迷った際に使用すると効果的です。

4. テレポートの戦略的使用：
   - 敵に囲まれた際の脱出手段として有効です。
   - 大きな壁を迂回する際にも使用できますが、消費するリソースとのバランスを考慮する必要があります。

5. 明るさ増加の使い分け：
   - 短期的な視界確保が必要な場合に有効ですが、MPを大量に消費するため慎重に使用する必要があります。
   - 敵が近くにいる可能性が高い場合や、重要なアイテムを探す際に使用すると効果的です。

### カスタマイズ方法

ゲームの設定は`maze_config.json`ファイルを通じてカスタマイズできます。主な設定項目は以下の通りです：

```json
{
    "WALL_COLOR": [
        0,
        0,
        0
    ],
    "ROUTE_COLOR": [
        255,
        255,
        255
    ],
    "UI_BACKGROUND_COLOR": [
        0,
        0,
        0
    ],
    "GOAL_COLOR": [
        0,
        255,
        0
    ],
    "PLAYER_COLOR": [
        30,
        144,
        255
    ],
    "ENEMY_COLOR": [
        128,
        0,
        32
    ],
    "GUAGE_BACKGROUND_COLOR": [
        30,
        30,
        30
    ],
    "MP_GAUGE_COLOR": [
        255,
        128,
        0
    ],
    "MP_GAUGE_LETTER_COLOR": [
        255,
        255,
        255
    ],
    "ITEM_COOLDOWN_GUAGE_COLOR": [
        124,
        252,
        0
    ],
    "SIGHT_GAUGE_COLOR": [
        121,
        205,
        255
    ],
    "SIGHT_GAUGE_LETTER_COLOR": [
        255,
        255,
        255
    ],
    "HINT_ARROW_COLOR": [
        255,
        243,
        176
    ],
    "MAX_MP": 20,
    "MAX_SIGHT": 2,
    "RESTORE_MP_PER_SECONDS": 2,
    "RESTORE_SIGHT_PER_SECONDS": 0.4,
    "HINT_DURATION": 3,
    "HINT_MP_COST": 10,
    "TELEPORT_MP_COST": 5,
    "TELEPORT_SIGHT_COST_PER_DISTANCE": 0.5,
    "MIN_SIGHT_FOR_TELEPORT": 1,
    "MP_FOR_BRIGHTNESS_VALUE_PER_SECOUNDS": 20,
    "MP_FOR_BRIGHTNESS_COST_PER_SECOUNDS": 5,
    "MP_FOR_BRIGHTNESS_DECAY_PER_SECOUNDS": 80,
    "TRANSPARENT_DURATION": 1,
    "MONSTER_ADDING_INTERVAL": 20,
    "VISIBLE_BORDER": 0.05,
    "CELL_SIZE": 20,
    "GAUGE_HEIGHT": 20,
    "GAUGE_MARGIN": 10,
    "ITEM_BOX_SIZE": 90,
    "ITEM_BOX_MARGIN": 10,
    "FPS": 30,
    "MAX_SIZE": 600,
    "SOUND_DIR": "sounds/",
    "FONT_DIR": "C:/Windows/Fonts",
    "FONT_NAME": "HGRGM.TTC",
    "FONTSIZE": 24,
    "GAUGE_FONTSIZE": 18,
    "ITEM_FONTSIZE": 20,
    "ACTION_LOG_DIR": "log/"
}
```

これらの値を変更することで、ゲームの難易度や見た目をカスタマイズできます。

設定をJSONファイルから読み込むには：

```python
game = MazeGame(maze, labels, start_goal_candidates)
if os.path.exists("maze_config.json"):
    game.load_config_from_json("maze_config.json")
```

現在の設定をJSONファイルに保存するには：

```python
game.save_config_to_json("maze_config.json")
```

#### <span style="color: #FF6B6B;">設定のカスタマイズ</span>

> **Note:** この特別なセクションを読むのはあなた次第。ゲームの本来の経験を楽しみたい方は、スキップしても大丈夫です。好奇心旺盛な方は、新しい遊び方を発見できるかもしれません。

`maze_config.json` ファイルを編集することで、ゲームの様々な要素をカスタマイズできます。以下は主な設定項目です：

- `CELL_SIZE`: セルのサイズ（ピクセル単位）
- `MAX_MP`: プレイヤーの最大MP
- `MAX_SIGHT`: プレイヤーの最大視界範囲
- `RESTORE_MP_PER_SECONDS`: 1秒あたりのMP回復量
- `RESTORE_SIGHT_PER_SECONDS`: 1秒あたりの視界回復量
- `HINT_MP_COST`: ヒント使用時のMPコスト
- `TELEPORT_MP_COST`: テレポート使用時のMPコスト
- `MONSTER_ADDING_INTERVAL`: 新しい敵が追加される間隔（秒）

注意: これらの値を変更すると、ゲームバランスに大きな影響を与える可能性があります。特に、`MAX_MP`、`MAX_SIGHT`、`RESTORE_MP_PER_SECONDS`、`MONSTER_ADDING_INTERVAL` などの値は、ゲームの難易度に直接関わります。

探究心旺盛なプレイヤーの方は、これらの設定を少しずつ調整してみると、思わぬ「発見」があるかもしれません。ただし、過度の調整はゲーム体験を損なう可能性があるので、慎重に行ってください。

開発者向けヒント: 設定値の組み合わせによっては、予想外の面白い効果が得られることがあります。例えば、`MAX_SIGHT` と `RESTORE_SIGHT_PER_SECONDS` の比率を変えると、ゲームの戦略が大きく変わるかもしれません。

## 依存ライブラリ

このプロジェクトは以下のライブラリを使用しています：

- numpy
- pygame
- matplotlib
- scipy
- opencv-python (cv2)

これらのライブラリは以下のコマンドでインストールできます：

```
pip install numpy pygame matplotlib scipy opencv-python
```

## プロジェクト構造

PythonDungeonMaker1/  
│  
├── DungeonMaker.py       # メインのゲームロジックと迷路生成アルゴリズム  
├── maze_config.json      # ゲーム設定ファイル  
├── README.md             # プロジェクトの説明書  
├── LICENSE               # ライセンス情報  
│  
├── sounds/               # ゲーム内で使用する音声ファイル  
│   ├── hint.mp3  
│   ├── goal.mp3  
│   └── ...  
│  
└── log/                  # ゲームプレイのログファイル保存ディレクトリ  

## 音声ファイルについて

**注意**: このリポジトリには、著作権上の理由から音声ファイルは含まれていません。

音声ファイルがなくても動作します。  
ゲームを完全に動作させるためには、以下の手順で音声ファイルを用意する必要があります：

1. `sounds/` ディレクトリを作成してください（存在しない場合）。

2. 以下の音声ファイルを `sounds/` ディレクトリに配置してください：
   - hint.mp3
   - goal.mp3
   - teleport.mp3
   - game_over.mp3
   - hit_enemy.mp3
   - light.mp3
   - light_end.mp3
   - monster_move.mp3
   - vision_monster.mp3
   - extra_light.mp3
   - pathfinder.mp3

3. 音声ファイルは、著作権に配慮して、自身で作成するか、適切なライセンスの下で提供されているものを使用してください。

4. [sounds/README.md](sounds/README.md) ファイルには、開発中に使用した音源の出典情報が記載されています。これらの情報は参考として提供されていますが、同じ音源を使用する必要はありません。

**音声ファイルが見つからない場合、ゲームは音なしで動作します**が、一部の機能（音声フィードバック）が制限される可能性があります。

推奨される音声ファイルのフォーマットはMP3ですが、pygameがサポートする他の形式（WAV、OGG等）も使用可能です。

## フォント設定

### Windows ユーザー

デフォルトでは、Windows向けにHGゴシックフォントが設定されています。この設定は `maze_config.json` ファイル内で確認できます。

### Mac ユーザー

Macでは、HGゴシックフォントがデフォルトで利用できないため、以下の手順で適切な日本語フォントを設定してください。

1. Mac互換の日本語フォントを選択します。以下のいずれかがおすすめです：
   - ヒラギノ角ゴシック（Hiragino Kaku Gothic ProN）
   - 游ゴシック（Yu Gothic）
   - osaka

2. `maze_config.json` ファイルを開き、以下の項目を変更します：

   ```json
   {
     "FONT_DIR": "/System/Library/Fonts",
     "FONT_NAME": "HiraginoSans-W3.otf"
   }
   ```

   注: FONT_NAME は選択したフォントのファイル名に合わせて変更してください。  
   選択したフォントが別の場所にある場合は、FONT_DIR も適切なパスに変更してください。変更を保存し、ゲームを再起動します。

### その他のOSユーザー

お使いのOSで利用可能な日本語フォントを選択し、上記の手順と同様に maze_config.json ファイルを編集してください。

### フォント設定に関するトラブルシューティング

フォントの設定に問題がある場合、ゲームは代替フォントを使用して起動を試みます。フォントの問題でゲームが起動しない場合は、以下を試してください：

- システムにインストールされている別の日本語フォントを指定する。
- 日本語フォントをインストールする（OSの指示に従ってください）。
- 一時的に英語フォントを使用する（日本語の表示が正しく行われない可能性があります）。

フォントの設定に関して問題が解決しない場合は、Issue を作成してサポートを求めてください。

## 使用方法

1. リポジトリをクローンまたはダウンロードします。
2. 必要なライブラリをインストールします。
3. `python DungeonMaker.py`を実行してゲームを開始します。

## 開発者向け情報

### コードの構造

プロジェクトの主要なクラスと役割：

- `Constant`：共通メソッドを管理
- `Analyzer`：迷路生成と難易度解析のアルゴリズムを実装
- `Enemy`：敵キャラクターの動作ロジックを管理
- `GameItem`：ゲーム内アイテムの基本クラス
- `PlayerStatus`：プレイヤーの状態を管理
- `MazeGame`：ゲームのメインロジックとUI処理を実装

### カスタマイズと拡張

1. 新しいアイテムの追加：
   - `GameItem`クラスを継承して新しいアイテムクラスを作成
   - `MazeGame.initialize_items()`メソッド内でアイテムを初期化

2. 敵の行動パターン追加：
   - `Enemy`クラスの`choice_direc()`メソッドに新しい行動パターンを実装

3. 新しい難易度解析手法の追加：
   - `Analyzer`クラスに新しいメソッドを追加
   - 既存の`difficulty()`メソッド内で新しい手法を組み込む

4. UIのカスタマイズ：
   - `MazeGame`クラス内の描画関連メソッドを修正

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENCE)ファイルを参照してください。

## 貢献

バグ報告や機能改善の提案は、GitHubのIssueを通じて行ってください。プルリクエストも歓迎します。

---

詳細な情報や実装の詳細については、ソースコードのコメントを参照してください。
