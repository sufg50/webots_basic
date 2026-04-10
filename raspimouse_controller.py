from controller import Robot

def run_robot():
    # ロボットのインスタンス化
    robot = Robot()
    
    # Webotsの基本時間単位（ms）を取得
    timestep = int(robot.getBasicTimeStep())
    
    # 2. モーターの取得 (PROTOファイル内の name と完全一致させる)
    left_motor = robot.getDevice('left_wheel_joint')
    right_motor = robot.getDevice('right_wheel_joint')
    
    # 3. モーターの初期設定 (速度制御モード)
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    
    # 初速度を設定
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    print("ラズパイマウス、起動成功")

    # 4. メインループ
    while robot.step(timestep) != -1:
        
        # シミュレーション開始からの経過時間（秒）を取得
        current_time = robot.getTime()
        
        # 最初の3秒間は直進
        if current_time < 3.0:
            left_motor.setVelocity(2.0)
            right_motor.setVelocity(2.0)
            
        # 3秒〜5秒の間は右タイヤを速くして左にカーブ
        elif current_time < 5.0:
            left_motor.setVelocity(1.0)
            right_motor.setVelocity(2.0)

        elif current_time < 8.0:
            left_motor.setVelocity(1.0)
            right_motor.setVelocity(2.5)

        # 5秒以降は再び直進
        else:
            left_motor.setVelocity(2.0)
            right_motor.setVelocity(2.0)
if __name__ == "__main__":
    run_robot()
