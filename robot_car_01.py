"""
Webotsの既存車両（Bmwx5）を複数のウェイポイントへ自動走行させるプログラム。
GPSから取得した現在位置と、直前位置との差分（移動ベクトル）から車両の進行方向（車の向き）を推定し、目的地（ウェイポイント）への方位角との差分を「向きのズレ」として算出。
ズレが小さくなる方向へステアリング（操縦角度）を比例制御（P制御）で与え、目標へ向かうようにステアリングを調整。
ウェイポイントに一定距離以内まで接近すると次のウェイポイントへ切り替え、リストを巡回しながら走行を継続します。
"""

import math
from vehicle import Driver
from controller import GPS

TIME_STEP = 60

driver = Driver()
driver.setSteeringAngle(0.0)
driver.setCruisingSpeed(20)

gps = driver.getDevice("gps")
gps.enable(TIME_STEP)

# 車の向きを覚えるための変数
prev_x, prev_y = 0.0, 0.0
car_angle = 0.0 

# ウェイポイント設定
targets = [ [-98, 24],[-107, -33], [-83, -100], [40, -95], [45, 5], [15, 35],[-54,45]] 

while driver.step() != -1:
    values = gps.getValues()
    if math.isnan(values[0]): continue
    
    x1, y1 = values[0], values[1]
    time = driver.getTime()

    # GPSの履歴から「今の車の向き（方角）」を計算
    if abs(x1 - prev_x) > 0.01: # 車両が1cm以上動いたら車の向きを更新
        car_angle = math.atan2(y1 - prev_y, x1 - prev_x)
        print(f"car_angle(ラジアン）: {car_angle}")
        print(f"car_angle(角度）: {math.degrees(car_angle)}")
        prev_x, prev_y = x1, y1

    if 'wp_idx' not in locals(): wp_idx = 0
    way_point = targets[wp_idx]

    # 目標への方角（ワールド座標での角度）。角度は公式「アークタンジェント＊（高さ/底辺）」で求められる。
    target_angle = math.atan2(way_point[1] - y1, way_point[0] - x1)
    
    print(f"target_angle(ラジアン）: {target_angle}")
    print(f"target_angle(角度）: {math.degrees(target_angle)}")
    
    # 車とウェイポイントの２点の距離を計算（弾同士の当たり判定と同じ理屈）
    distance = math.sqrt((x1 - way_point[0])**2 + (y1 - way_point[1])**2)
    
    # 次のウェイポイントに向かう
    if distance < 5:
        wp_idx = (wp_idx + 1) % len(targets)

    # 「目標の方角」から「今の車の向き」を引いて、ズレを出す
    # この diff が 0 になれば、車は目標を真っ直ぐ向いていることになります
    diff = target_angle - car_angle
    driver.setSteeringAngle(0.5 * -diff)