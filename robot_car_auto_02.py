"""
Webotsの既存車両（Bmwx5）を複数のウェイポイントへ自動走行させるプログラム。
車両は、カメラで道路の黄色い線を見つけて追従しながら走行します。

具体的なカメラの処理アルゴリズムは、
1. カメラ画像の下半分をスキャンして、黄色い線を見つけます。
2. 黄色い線の位置をもとに、車がどれくらいハンドルを切るべきか（操舵角）を計算します。

LIDAR（レーザースキャナー）を使って前方の障害物を検知し、障害物が近づいてきたら回避行動を取ります。
具体的なLiDARの処理アルゴリズムは、
1. 正面の一定範囲（例：左右20度）だけをスキャンして、障害物を見つけます。
2. 障害物が見つかったら、その障害物の位置（角度）と距離を計算します。

run1とrun2の違いとしては、
run1ではif文によってハンドルの切り替え向きをしていたが、
run2ではポテンシャル法によってハンドルの切り替えを合成している。

"""
import cv2             # 画像処理のための強力なライブラリ「OpenCV」を読み込みます。
import numpy as np     # 数値計算や配列（画像のピクセルデータなど）を高速に扱うためのライブラリ「NumPy」を読み込みます。
import math
from vehicle import Driver  # Webotsの自動車専用の機能（アクセルやハンドル操作）を使うための設計図を読み込みます。
from controller import GPS, Node # GPS（位置情報）などのセンサを使うための機能を読み込みます。

""" 
ロボットカークラス 
プログラム全体を「車の設計図（クラス）」としてまとめています。
"""
class RobotCar():
    # --- クラス変数（この車全体で共通して使う基本設定） ---
    SPEED   = 60        # 車が走る基本スピードを設定します。
    UNKNOWN = 99999.99  # カメラで黄色い線が全く見つからなかった時に、エラーの目印として使う異常な数値です。
    FILTER_SIZE = 3     # 黄色ライン用のフィルタ
    TIME_STEP   = 30    # センサ（カメラやGPS）のデータを取得する間隔です。60[ms]（1秒間に約16回）ごとに目を開いて景色を見ます。
    CAR_WIDTH   = 2.015 # 車幅[m]
    CAR_LENGTH  = 5.0   # 車長[m]    
    """ 
    コンストラクタ（初期化処理）
    プログラムがスタートして「車が生まれた瞬間」に1回だけ実行される準備運動です。
    """
    def __init__(self):  
        # 1. ウエルカムメッセージを表示する関数を呼び出します。
        self.welcomeMessage()    
        
        # 2. 車の運転手（Driver）を呼び出し、初期設定をします。
        self.driver = Driver()                   # 車を操作するコントローラーの本体を作ります。
        self.driver.setSteeringAngle(0)          # 最初のハンドルの角度を 0（まっすぐ）にします。
        self.driver.setCruisingSpeed(self.SPEED) # アクセルを踏んで、スピードを20km/hに設定します。
        self.driver.setDippedBeams(True) # ヘッドライト転倒
        
        # LIDAR (SICK LMS 291)
        self.lidar = self.driver.getDevice("Sick LMS 291")
        self.lidar.enable(self.TIME_STEP)
        self.lidar_width = self.lidar.getHorizontalResolution()
        self.lidar_range = self.lidar.getMaxRange()
        self.lidar_fov   = self.lidar.getFov();  
        
        # Liderの性能をコンソール（画面下の黒い部分）に表示して確認します。
        print("lidar: width=%d max_range=%d fov=%g" % \
            (self.lidar_width, self.lidar_range, self.lidar_fov))
        
        # 3. GPS（カーナビ）の準備
        self.gps = self.driver.getDevice("gps")  # 車に付いている "gps" という名前の装置を取得します。
        self.gps.enable(self.TIME_STEP)          # 60msごとにGPSが自分の位置を測るように電源を入れます。

        # 4. カメラ（目）の準備
        self.camera = self.driver.getDevice("camera") # "camera" という名前の装置を取得します。
        self.camera.enable(self.TIME_STEP)            # 60msごとに景色を撮影するように電源を入れます。
        
        # カメラのスペック（性能）を調べて、車自身のデータとして記憶しておきます。
        self.camera_width  = self.camera.getWidth()   # 撮影する画像の横幅（例：128ピクセル）
        self.camera_height = self.camera.getHeight()  # 撮影する画像の縦幅（例：64ピクセル）
        self.camera_fov    = self.camera.getFov()     # カメラの視野角（どれくらい広く見えるか）
        
        # カメラの性能をコンソール（画面下の黒い部分）に表示して確認します。
        print("camera: width=%d height=%d fov=%g" % \
            (self.camera_width, self.camera_height, self.camera_fov))

        # 5. ディスプレイ（カメラ映像を映し出すモニター）の準備
        self.display = self.driver.getDevice("display") # "display" という名前の装置を取得します。
        self.display.attachCamera(self.camera)          # モニターにカメラの映像を接続して映し出します。
        self.display.setColor(0xFF0000)                 # モニターの文字色などを赤色(RGBのRがMAX)に設定します。
        
        # 6.移動平均用の過去のずれ
        self.history = []
        
        # 7.PID制御用
        self.prev_error = 0.0 # 【D制御用】1コマ前の「ズレ」を記憶するメモ帳
        
        self.integral = 0.0   # 【I制御用】過去のズレの「積み重ね（合計）」
        
    """ ウエルカムメッセージ """    
    def welcomeMessage(self):
        # プログラム開始時にコンソールに文字を表示します。
        print("*********************************************")
        print("* Welcome to a simple robot car program   *")
        print("*********************************************")       

    """ 
    移動平均フィルタ（データのふらつきを滑らかにする機能）
    ※現在ここは空っぽで、入ってきた数値をそのまま返しています。
    　もし車がガタガタふらつくなら、ここに「過去数回分のハンドルの角度の平均を計算する」処理を書くとスムーズになります。
    """
    def maFilter(self, new_value):
        N = 5  # 過去何回分の平均をとるか（この数字を変えると滑らかさが変わります）
        
        # 1. 新しいハンドルの角度をメモ帳の一番最後に追加する
        self.history.append(new_value)
        
        # 2. もしメモ帳の記録が N回 を超えたら、一番古い記録（0番目）を消しゴムで消す
        if len(self.history) > N:
            self.history.pop(0)
            
        # 3. メモ帳に残っている直近 N回分 の合計を出し、記録の数で割って平均を出す
        average = sum(self.history) / len(self.history)
        
        return new_value # 計算した平均値を、最終的なハンドルの角度として返す

    """ 
    ステアリングの制御（ハンドルの安全装置）
    計算したハンドルの角度（steering_angle）を実際のタイヤに伝えます。
    ステアリングの制御： wheel_angleが正だと右折，負だと左折
    """
    def control(self, steering_angle):
        LIMIT_ANGLE = 0.5 # ハンドルを切る限界の角度を 0.5ラジアン（約28度）に決めます。
        
        # もし計算結果が「右に急カーブしすぎ！」だったら、限界の0.5で止めます。
        if steering_angle > LIMIT_ANGLE:
            steering_angle = LIMIT_ANGLE
        # もし計算結果が「左に急カーブしすぎ！」だったら、限界の-0.5で止めます。
        elif steering_angle < -LIMIT_ANGLE:
            steering_angle = -LIMIT_ANGLE
            
        # 安全確認が終わった角度を、実際のタイヤのモーターに送って曲がらせます。
        self.driver.setSteeringAngle(steering_angle)
    
    """ 
    画素と黄色の差の平均を計算（これは黄色か？の判定テスト）
    カメラで見ている1つの点（ピクセル）の色が、ターゲットの黄色とどれくらい似ているかを点数（ズレの大きさ）で返します。
    """
    def colorDiff(self, pixel, yellow):
        d, diff = 0, 0
        # 画像の色は B(青)・G(緑)・R(赤) の3つの数字の組み合わせでできています。これを順番にチェックします。
        for i in range (0,3):
            # int()で囲むことで、マイナスの数字になってもエラー（オーバーフロー）が起きないようにします。
            # 例: ピクセルが[100, 100, 100]で黄色が[95, 187, 203]なら、差の絶対値（ズレ）を計算します。
            d = abs(int(pixel[i]) - int(yellow[i]))
            diff += d  # B, G, R それぞれのズレを合計します。
            
        # 3色のズレの平均値を計算して返します。この数字が小さいほど「ターゲットの黄色とそっくり」という意味になります。
        return diff/3


    """ 
    黄色ラインを追従するための操舵角の計算（脳みそ）
    カメラの画像全体を調べて黄色い線を見つけ出し、どれくらいハンドルを切ればいいか（操舵角）を計算します。
    """
    def calcSteeringAngle(self, image):
        YELLOW = [95, 187, 203]   # 探したい黄色の値です。(青=95, 緑=187, 赤=203)
        sumx = 0                  # 見つけた黄色の点の「X座標（横の位置）」を全て足し算するための箱です。
        pixel_count = 0           # 黄色い点を「何個」見つけたかを数えるカウンターです。
        
        # 【ステップ1】画像のどこを探すか？
        # 空や遠くの景色を探しても無駄なので、画像の「上から1/3より下（道路がある場所）」だけをスキャンします。
        # yは縦（高さ）、xは横（幅）を表します。
        for y in range(int(1.0*self.camera_height/3.0), self.camera_height):
            for x in range(0, self.camera_width):
                
                # 【ステップ2】黄色かどうかの判定
                # さっき作った colorDiff 関数で色のズレを測り、ズレが「30未満」なら黄色だと認定します！
                if self.colorDiff(image[y,x], YELLOW) < 30: 
                    sumx += x         # 見つけた点のX座標（横の位置）を足し算します。
                    pixel_count += 1  # 見つけた個数を1つ増やします。
   
        # 【ステップ3】ハンドルの角度の計算
        # もし黄色い点が1個も見つからなかったら、異常事態（UNKNOWN）を返して報告します。
        if pixel_count == 0:
            return self.UNKNOWN
            
        # 黄色い点が見つかった場合の計算ルート
        else:
            
            # ターゲット位置（0.0〜1.0）。0.25なら「画面の左から25%の位置に線が来るように走る」
            # 0.5なら中央を走る
            TARGET_POS = 0.25
            
            # 比例制御
            # 見つけた黄色の全X座標の合計（sumx）を、(黄色い点の個数 × 画面の横幅) で割ります。
            # これにより、「画面の左端を 0.0、右端を 1.0 とした時、黄色い線の中心はどの割合の位置にあるか？」が求まります。
            # 例：横幅100の画面で、中心が左から70番目にあれば、y_ave は 0.7 になります。

            # わかりやすいように数式を合体せず記載する
            # 1. まず、見つけた黄色いピクセルの「平均のX座標（重心）」を計算する
            # center_pixel_x = float(sumx) / pixel_count 
            
            # 2. その重心が、画面全体の横幅に対して「どの割合（0.0〜1.0）の位置にあるか」を計算する
            # y_ave = center_pixel_x / self.camera_width

            
            # 3.画面のど真ん中は「0.5」です。(y_ave - 0.5) を計算すると、真ん中からのズレがわかります。
            # 例：0.7 - 0.5 = 0.2。つまりプラス（右側）にズレているから、右にハンドルを切ろう！と計算します。
            # 最後にカメラの視野角（fov）を掛けて、実際のステアリングの角度に変換します。
            # 広角レンズの場合、現実世界より黄色の線の相対的な位置の値が小さくなるから、視野角をかける
            # steer_angle = (y_ave - TARGET_POS) * self.camera_fov
            
            
            
            
            # PID制御用
            y_ave = float(sumx) /(pixel_count * self.camera_width) 
            
            # 1. 【P制御の素】今のズレを計算する
            error = y_ave - TARGET_POS
            
            # 2. 【I制御の素】ズレの積み重ね（積分）を計算する ＝ (これまでの合計 + 今のズレ)
            self.integral += error
            
            # 3. 【D制御の素】ズレの変化スピード（微分）を計算する ＝ (今のズレ - 1コマ前のズレ)
            diff_error = error - self.prev_error
            
            # 4. 効き目の強さ（ゲイン）を設定する
            # D制御の強さ（未来予測ブレーキの強さ。猛スピードで白線に近づいた時に、行き過ぎないようあえて逆ハンドルを切るための力）
            Kd = 2.0 
            # I制御の強さ（蓄積されたズレを直す力。※足し算で巨大な数字になるので、とても小さな値をかけます） 
            Ki = 0.01
            
            # 5. P、I、D すべてを足し合わせて、最終的なハンドルの角度を決める！
            steer_angle = ((y_ave - TARGET_POS) * self.camera_fov) + (Ki * self.integral) + (Kd * diff_error)
            
            # 6. 次の計算（1コマ後）のために、今のズレをメモ帳に書き残しておく
            self.prev_error = error
            
            return steer_angle  # 計算した「ハンドルを切る角度」を返します。

    """ 障害物の方位と距離を返す. 障害物を発見できないときはUNKNOWNを返す．
        ロボットカー正面の矩形領域に障害物がある検出する    
    """
    def calcObstacleAngleDist(self, lidar_data):
            # --- 【準備】ルールの設定 ---
            OBSTACLE_HALF_ANGLE = 20.0 # 正面から左右にどれくらい（インデックス幅）スキャンするか
            OBSTACLE_DIST_MAX   = 20.0 # これより遠いもの（20m以上）は無視する（止まらなくていい）
            OBSTACLE_MARGIN     = 0.1  # 車幅にプラスする「安全マージン（横の隙間）」10センチ
            
            sumx = 0            # 障害物に当たったレーザーの「角度（インデックス）」の合計
            collision_count = 0 # 障害物に当たったレーザーの「本数」
            obstacle_dist   = 0.0 # 障害物までの「距離」の合計
    
            # --- 【ステップ1】正面だけを調べる ---
            # 360度すべて調べると横や後ろの壁に反応してしまうので、「真正面（lidar_width/2）」を中心にして、
            # 左右20本分（OBSTACLE_HALF_ANGLE）のレーザーだけをチェックします。
            # ※左右20本というのは、角度ではない。liderの正面が0度となる。
            # lidar: width=180 max_range=80 fov=3.14159なので、
            # self.lidar_width/2 - OBSTACLE_HALF_ANGLEは、180/2-20で70,
            # self.lidar_width/2 + OBSTACLE_HALF_ANGLEは、180/2+20で110
            for i in range(int(self.lidar_width/2 - OBSTACLE_HALF_ANGLE), \
                int(self.lidar_width/2 + OBSTACLE_HALF_ANGLE)):
                
                dist = lidar_data[i] # i番目のレーザーが測った距離
                
                # --- 【ステップ2】近い障害物をピックアップ ---
                # もし距離が20m未満（遠すぎる壁や空ではない）なら、障害物としてカウント！
                if dist < OBSTACLE_DIST_MAX: 
                    sumx += i               # カメラの時と同じ！見つけた場所（インデックス）を足す
                    collision_count += 1    # 当たったレーザーの本数を増やす
                    obstacle_dist += dist   # 距離を足していく
                    
            # --- 【ステップ3】障害物がない場合の処理 ---
            if collision_count == 0 or obstacle_dist > collision_count * OBSTACLE_DIST_MAX:  
                return self.UNKNOWN, self.UNKNOWN # 何もなかったので、UNKNOWN（見つからず）を返す
                
            # --- 【ステップ4】障害物の位置と距離を計算 ---
            else:
                # 障害物の「角度」を計算（※カメラで黄色い線の重心を求めたのと全く同じ計算式です！）
                # obstacle_angle = (float(sumx) /(collision_count * self.lidar_width) - 0.5) * self.lidar_fov                

                # わかりやすいように数式を合体せず記載する
                # 1. まず、障害物に当たったレーザーの「平均の番号（重心インデックス）」を計算する
                # 例：80番と90番と100番のレーザーが当たったら、平均は90番になります。
                center_ray_index = float(sumx) / collision_count 
                
                # 2. その平均の番号が、レーザー全体の総本数（180本）に対して「どの割合（0.0〜1.0）の位置にあるか」を計算する
                # 例：平均が90番なら、90 / 180 = 0.5（ちょうど真ん中）になります。
                ray_ratio = center_ray_index / self.lidar_width
                
                # 3. 真正面（0.5）からどれくらいズレているかを計算し、最後に視野角（fov）を掛けて実際の角度（ラジアン）に変換する
                # 例：ray_ratioが0.5なら、(0.5 - 0.5) * fov = 0.0ラジアン（真正面にある）と計算されます。
                obstacle_angle = (ray_ratio - 0.5) * self.lidar_fov
                
                # 障害物までの「距離」の平均を計算（合計距離 ÷ 当たった本数）
                # LiDARは車の中心にあるため、障害物までの直線距離であるが、斜めとなる。
                obstacle_dist  = obstacle_dist/collision_count
                
                # --- 【ステップ5】本当にぶつかるかどうかの最終判定（三角関数） ---
                # ここが一番の山場！斜め前にある障害物が、自分の車の幅（車線）に被っているかを判定します。   
                # 1.「まっすぐ前」を見た状態から、障害物を見るために首を横にどれくらい（何度）回したか？ —— その首を回した角度こそがθ
                #   -0.5をしているから、正面を0度としたときのずれ
                # 2.斜辺はliderから障害物までの直線
                # 3.三平方の定理から「斜辺×sinΘ=高さ」なのでこれが車幅以下ならば衝突しているとみなす
                # 4.Pythonのmath.sin(x)は引数xにラジアンをいれる。
                # つまり、0.25の横ずれで視野角が3.14のとき、0.785 ラジアンとなり、現実のずれは４５度となるけれど、
                # 0.25のずれであっても、視野角が1.0のときは現実のずれは、0.25ラジアン（約14度）ということ。
                # 5.なぜ割合がラジアンになるかというと、分度器で考える場合分度器の真ん中は90度、）
                #  分度器の「全体の 25%（0.25）]は45度となり、
                #  「全体に対する割合（比率）」は、「全体の角度に対する割合」と完全に一致（比例）する
                if abs(obstacle_dist * math.sin(obstacle_angle)) < 0.5 * self.CAR_WIDTH + OBSTACLE_MARGIN:
                    return obstacle_angle, obstacle_dist # 被っている＝衝突する！
                else:
                    return self.UNKNOWN, self.UNKNOWN    # 被っていない＝横を通り抜けられる！
    """ 
    実行（メインループ）
    車が走っている間、ずっと「景色を見る→考える→ハンドルを切る」を繰り返す心臓部です。
    """

    def run1(self): 
        step = -1
        avoid_timer = 0 # 障害物を見つけたときに連続してよけ続ける
        # シミュレーションが動いている限り、永遠にこの while ループの中をぐるぐる回り続けます。
        while self.driver.step() != -1:
            step +=1 
            BASIC_TIME_STEP = 10 # シミュレータの「1コマ」は10[ms]です。
            
            # センサの更新間隔（TIME_STEP=30）ごとに1回だけ、以下の処理を行います。（つまり3コマに1回のペース）
            if step % int(self.TIME_STEP / BASIC_TIME_STEP) == 0:

                # 1. GPSデータの取得（現在は取得するだけで使っていません）    
                values = self.gps.getValues()       

                # 2. 目を開けて景色を見る（カメラ画像の取得）
                camera_image = self.camera.getImage()
                
                # カメラから届いたデータはコンピュータが読みにくい暗号のような形なので、
                # OpenCV（画像処理ライブラリ）が計算しやすい「3次元の配列（縦×横×色）」に変換します。
                cv_image = np.frombuffer(camera_image, np.uint8).reshape((self.camera_height, self.camera_width, 4))


                # 障害物回避
                # Lidar
                lidar_data = self.lidar.getRangeImage()  
                obstacle_angle, obstacle_dist = self.calcObstacleAngleDist(lidar_data)
                STOP_DIST = 5 # 停止距離[m]

                if obstacle_dist < STOP_DIST:
                    print("%d:Find obstacles(angle=%g, dist=%g)" % \
                        (step, obstacle_angle, obstacle_dist))
                    avoid_timer = 10
                else:
                    # 【追加】もし障害物がなくなったらストップを解除する
                    stop = False    
                # 3. 脳みそで考える（操舵角の計算）
                # 変換した画像を calcSteeringAngle に渡して、ハンドルの角度を計算させます。
                # さらに、その角度を maFilter（移動平均）に通して滑らかにします。
                steering_angle = self.maFilter(self.calcSteeringAngle(cv_image))
                
                # 4. 手足を動かす（ハンドルの操作）
                # 【モード1：回避モード】タイマーが0より大きい間は、絶対によけ続ける！
                if avoid_timer > 0:
                    avoid_timer -= 1 # 1コマ進むごとにタイマーを1減らす
                    print("%d: 障害物回避中！(残りタイマー: %d)" % (step, avoid_timer))
                    self.driver.setCruisingSpeed(self.SPEED * 0.5) 

                    # 障害物が自分の右側(プラス)にあるなら左(マイナス)へ、左側なら右へ逃げる
                    if obstacle_angle >= 0:
                        direction = -0.5 # 左へ逃げる
                    else:
                        direction = 0.5  # 右へ逃げる
                        
                    if avoid_timer < 4:
                        self.control(-direction)
                    else:
                        self.control(direction)
                    
                # 黄色い線が見つかっている場合
                elif steering_angle != self.UNKNOWN:
                    print("%d:Find the yellow line" % step)   # コンソールに「線を見つけた！」と表示
                    
                    if abs(steering_angle) > 0.05:
                        self.driver.setCruisingSpeed(self.SPEED * 0.5) # カーブのときは減速する
                    else:
                        self.driver.setCruisingSpeed(self.SPEED)  # アクセルを踏んでスピードを維持します
                    self.control(steering_angle)              # 計算した角度の通りにハンドルを切ります
                    
                # 黄色い線を見失っている場合
                else: 
                    print("%d:Lost the yellow line" % step)   # コンソールに「見失った！」と表示
                    self.driver.setCruisingSpeed(0)           # 危険なので、スピードを0にして急ブレーキをかけます！

    """ 
    実行（メインループ）
    車が走っている間、ずっと「景色を見る→考える→ハンドルを切る」を繰り返す心臓部です。
    """
    def run2(self): 
        step = -1        
        # 【追加】ポテンシャル法の「記憶」用の変数
        memory_timer = 0
        # シミュレーションが動いている限り、永遠にこの while ループの中をぐるぐる回り続けます。
        while self.driver.step() != -1:
            step +=1 
            BASIC_TIME_STEP = 10 # シミュレータの「1コマ」は10[ms]です。
            
            # センサの更新間隔（TIME_STEP=30）ごとに1回だけ、以下の処理を行います。（つまり3コマに1回のペース）
            if step % int(self.TIME_STEP / BASIC_TIME_STEP) == 0:

                # 1. GPSデータの取得（現在は取得するだけで使っていません）    
                values = self.gps.getValues()       

                # 2. 目を開けて景色を見る（カメラ画像の取得）
                camera_image = self.camera.getImage()
                cv_image = np.frombuffer(camera_image, np.uint8).reshape((self.camera_height, self.camera_width, 4))

                # 障害物回避 (Lidar)
                lidar_data = self.lidar.getRangeImage()  
                obstacle_angle, obstacle_dist = self.calcObstacleAngleDist(lidar_data)


                # ==========================================================
                # 3. 脳みそで考える（ポテンシャル法による操舵角の計算）
                # ==========================================================
                
                # 【元のコードをそのまま利用】カメラ画像から、黄色い線を追うための角度を計算
                camera_steering = self.maFilter(self.calcSteeringAngle(cv_image))

                # ① 引力（黄色い線へ向かう力）
                if camera_steering != self.UNKNOWN:
                    attractive_steer = camera_steering # 線が見えていれば、そのPIDの角度がそのまま「引力」になる
                else:
                    # 線を見失った場合は、右側通行なので「左側」に線があるはず。
                    # 探すために、ゆっくり左(-0.1)へ引っ張られる引力を設定しておく。
                    attractive_steer = -0.3

                # ② 斥力（障害物から逃げる力）
                repulsive_steer = 0.0
                OBS_AVOID_DIST = 10.0 # 障害物の10m以内に近づいたら反発力を発生させる
                
                if obstacle_dist != self.UNKNOWN and obstacle_dist < OBS_AVOID_DIST:
                    K_REP = 3.0 # 反発力の強さ（ゲイン）。大きくすると遠くから大きく避けます。
                    
                    # 障害物が自分の右側(プラス)にあるなら左(マイナス)へ、左側なら右へ逃げる
                    if obstacle_angle >= 0:
                        direction = -0.5 # 左へ逃げる
                    else:
                        direction = 0.5  # 右へ逃げる

                    # 【ポテンシャル法の要】距離が近いほど反発力が強くなる計算式： K * (1 / 障害物との距離)
                    safe_dist = max(obstacle_dist, 0.1)
                    repulsive_steer = direction * K_REP * (1.0 / safe_dist)

                    # 【追加】今計算した斥力を「記憶」し、タイマーをセットする！
                    # （約1.5秒間は、視界から消えてもこの反発力を信じる）
                    repulsive_memory = repulsive_steer
                    memory_timer = 50
                    
                elif memory_timer > 0:
                    memory_timer -= 1 # 時間を減らす
                    repulsive_steer = repulsive_memory # 記憶しておいた斥力を使い続ける！
                    print(" -> 視界ロスト！しかし斥力を記憶して回避継続中... (残: %d)" % memory_timer)

                # ③ 力の合成（引力 ＋ 斥力）
                # 最終的なハンドルの角度は、この2つの力を足し算するだけで決まる！
                total_steer = attractive_steer + repulsive_steer


                # ==========================================================
                # 4. 手足を動かす（ハンドルの操作とアクセル）
                # ==========================================================
                
                # コンソールで「引力と斥力の綱引き」の様子を観察できるように表示します
                print("%d: APF制御中 (引力: %.2f, 斥力: %.2f, 合計: %.2f)" % \
                      (step, attractive_steer, repulsive_steer, total_steer))
                
                # 急カーブ（障害物に近くて反発力が強い時など）は安全のために減速する
                if abs(total_steer) > 0.1:
                    self.driver.setCruisingSpeed(self.SPEED * 0.5)
                else:
                    self.driver.setCruisingSpeed(self.SPEED)
                
                # 合成した力の通りにハンドルを切る！
                # （※万が一計算結果が大きすぎても、controlメソッド内の LIMIT_ANGLE が安全に制限してくれます）
                self.control(total_steer)

""" 
メイン関数
プログラムが一番最初に実行するところです。
""" 
def main():  
    robot_car = RobotCar() # 「RobotCar」という設計図をもとに、実体の車（インスタンス）を1台生み出します。
    # robot_car.run1()        # if文による障害物回避
    robot_car.run2()      # ポテンシャル法による障害物回避
    
""" 
このスクリプトを直接実行した時のおまじない
（他のファイルから読み込まれた時は勝手に走らないようにするためのルールです）
"""
if __name__ == '__main__':
    main()