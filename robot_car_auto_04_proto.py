"""
このコードは、車の位置と向きを自力で計算し、ウェイポイントに向かって走るようにするものです。
【変更点1】車の位置と向きを覚えるための変数を用意します。
【変更点2】車の速度とハンドルの角度を取得して、車の位置と向きを更新するための計算を行います。
【変更点3】アッカーマンの数式で、x1, y1, car_angle を自力で計算（更新）します。
【変更点4】車とウェイポイントの距離を計算して、一定の距離以下になったら次のウェイポイントに向かうようにします。
"""

import math
from vehicle import Driver

TIME_STEP = 60
WHEELBASE = 2.995  # PROTOファイルから取得したホイールベース[m]
TRACK_REAR = 1.628 # 左右の後輪の間隔[m] （追加）
driver = Driver()
driver.setSteeringAngle(0.0)
driver.setCruisingSpeed(10)


# 車の向きを覚えるための変数
x1, y1 = -45.0, 45.88
car_angle = 3.14159
last_time = driver.getTime() # 最初の時間を記録しておく

# ウェイポイント設定
targets = [ [-95, 25],[-107, -33], [-83, -100], [40, -95], [45, 5], [15, 35],[-54,45]] 

while driver.step() != -1:

    # GPSの代わりに、車の速度とハンドルの角度を取得します
    speed_kmh = driver.getCurrentSpeed()
    steering = driver.getSteeringAngle()
    
    # 【追加】センサーが準備できていない(NaNの)最初の数フレームは計算をスキップ
    if math.isnan(speed_kmh) or math.isnan(steering):
        continue
    
    # 以降は先ほどと同じ
    speed_ms = speed_kmh / 3.6  # km/h から m/s に変換
    # === 1. シミュレータの正確な時計から dt を計算する ===
    current_time = driver.getTime()
    dt = current_time - last_time
    last_time = current_time

    # 変更点3：アッカーマンの数式で、x1, y1, car_angle を自力で計算（更新）します
    car_angle -= (speed_ms * math.tan(steering) / WHEELBASE) * dt
    x1 += speed_ms * math.cos(car_angle) * dt
    y1 += speed_ms * math.sin(car_angle) * dt

    print(f"posion:{x1,y1}")

    if 'wp_idx' not in locals(): wp_idx = 0
    way_point = targets[wp_idx]

    # 目標への方角（ワールド座標での角度）。角度は公式「アークタンジェント＊（高さ/底辺）」で求められる。
    target_angle = math.atan2(way_point[1] - y1, way_point[0] - x1)
    
    print(f"target_angle(ラジアン）: {target_angle}")
    print(f"target_angle(角度）: {math.degrees(target_angle)}")
    
    # 車とウェイポイントの２点の距離を計算（弾同士の当たり判定と同じ理屈）
    distance = math.sqrt((x1 - way_point[0])**2 + (y1 - way_point[1])**2)
    
    print(f"distance:{distance}")
    # 次のウェイポイントに向かう
    if distance < 5:
        wp_idx = (wp_idx + 1) % len(targets)

    # 「目標の方角」から「今の車の向き」を引いて、ズレを出す
    # この diff が 0 になれば、車は目標を真っ直ぐ向いていることになります
    diff = target_angle - car_angle
    
    # 「角度の正規化」を行う-3.14から3.14の範囲にする
    while diff > math.pi: diff -= 2.0 * math.pi
    while diff < -math.pi: diff += 2.0 * math.pi
    print(f"diff:{diff}")
    driver.setSteeringAngle(0.5 * -diff)