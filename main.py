import cv2 as cv
import numpy as np
import math

def detect_chessboard_corners(video_path, chessboard_size=(9, 6), square_size=1.0,
                              display_delay=300, max_frames=20, frame_interval=0.5, target_size=None):
    """
    체스보드 영상에서 코너 검출 함수
    - video_path: 영상 파일 경로
    - chessboard_size: 체스보드 내부 코너 개수 (가로, 세로)
    - square_size: 각 정사각형의 실제 크기
    - display_delay: 검출 결과 보여줄 때 지연 시간 (ms)
    - max_frames: 캘리브레이션에 사용할 최대 프레임 수
    - frame_interval: 프레임 수집 간 최소 시간 간격 (초)
    - target_size: (width, height) 형태의 영상 크기 (None이면 원본 사용)
    
    반환:
      objpoints: 실제 3D 객체 포인트 리스트
      imgpoints: 영상상의 2D 코너 포인트 리스트
      image_size: 영상 크기 (width, height)
    """
    # 체스보드의 3D 객체 포인트 (z=0 평면)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 실제 3D 포인트 모음
    imgpoints = []  # 영상상의 2D 포인트 모음
    
    # 코너 서브픽셀 보정 기준
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    frame_count = 0
    detected_frames = 0
    image_size = None
    last_capture_time = -frame_interval

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # target_size가 지정되어 있으면 프레임 리사이즈
        if target_size is not None:
            frame = cv.resize(frame, target_size)
        frame_count += 1
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        
        # 현재 프레임의 시간(초)
        time_stamp = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
        if time_stamp - last_capture_time < frame_interval:
            continue
        last_capture_time = time_stamp

        # 체스보드 코너 검출
        found, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        if found:
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, chessboard_size, corners_refined, found)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            detected_frames += 1

        if detected_frames >= max_frames:
            break

    cap.release()
    cv.destroyAllWindows()
    return objpoints, imgpoints, image_size

def calibrate_camera(objpoints, imgpoints, image_size):
    """
    수집한 객체 포인트와 이미지 포인트를 이용하여 카메라 캘리브레이션 수행
    반환:
      ret          : RMS 재투영 오차
      camera_matrix: 캘리브레이션된 카메라 행렬 (K)
      dist_coeffs  : 왜곡 계수
      rvecs        : 각 프레임별 회전 벡터
      tvecs        : 각 프레임별 평행 이동 벡터
    """
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    
    # print("캘리브레이션 결과:")
    # print("RMS 재투영 오차:", ret)
    # print("카메라 행렬 (K):\n", camera_matrix)
    # print("왜곡 계수:\n", dist_coeffs)
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs

def generate_extruded_heart_shape_points(n_points=100, scale=0.1, center=(4.0, 3.0, 0.0), depth=0.2):
    """
    하트 모양의 3D 점들 생성 (Extruded 형태)
    - n_points: 전면/후면의 점 개수 (동일한 개수)
    - scale: 하트 크기 조절
    - center: 하트 중심 (체스보드 좌표계 기준)
    - depth: extrusion 깊이 (전면은 z=0, 후면은 z=-depth)
    반환:
      front_points: 전면(Front) 점 배열 (n_points x 3)
      back_points : 후면(Back) 점 배열 (n_points x 3)
    """
    front_points = []
    back_points = []
    for i in range(n_points):
        t = 2 * math.pi * i / n_points
        # 하트 곡선 방정식
        x = -16 * (math.sin(t) ** 3)
        y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
        # 스케일 및 중심 이동
        x = x * scale + center[0]
        y = y * scale + center[1]
        front_points.append([x, y, center[2]])         # 전면은 z=0
        back_points.append([x, y, center[2] - depth])    # 후면은 -depth 만큼 내려감
    return np.array(front_points, dtype=np.float32), np.array(back_points, dtype=np.float32)

def init_video_writer(output_path, fps, frame_size, codec='XVID'):
    """
    VideoWriter 객체를 초기화하여 반환하는 함수
    - output_path: 저장할 영상 파일 경로
    - fps: 프레임 속도
    - frame_size: (width, height) 형태의 프레임 크기
    - codec: 코덱 문자열 (기본 'XVID')
    """
    fourcc = cv.VideoWriter_fourcc(*codec)
    writer = cv.VideoWriter(output_path, fourcc, fps, frame_size)
    return writer

def run_heart_ar(video_path, chessboard_size, square_size,
                 camera_matrix, dist_coeffs, target_size=None):
    """
    체스보드 영상에서 체스보드 코너 검출 후 cv.solvePnP로 카메라 자세 추정하고,
    미리 생성한 3D Extruded 하트 모양 AR 오브젝트(전면, 후면, 연결선)를 투영하여 영상에 오버레이.
    - video_path: 영상 파일 경로
    - chessboard_size: 체스보드 내부 코너 개수 (가로, 세로)
    - square_size: 체스보드 정사각형의 실제 크기
    - camera_matrix, dist_coeffs: 캘리브레이션 결과
    - target_size: (width, height) 형태의 영상 크기 지정 (None이면 원본 사용)
    """
    # 체스보드 객체 포인트 (z=0 평면)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 3D Extruded 하트 모양 생성: front (전면)와 back (후면)
    front_points_3d, back_points_3d = generate_extruded_heart_shape_points(n_points=100, scale=0.1, center=(4.0, 3.0, 0.0), depth=3)
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    # 출력 영상 크기 설정
    if target_size is not None:
        width, height = target_size
    else:
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    fps = cap.get(cv.CAP_PROP_FPS)
    writer = init_video_writer("demo.avi", fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if target_size is not None:
            frame = cv.resize(frame, target_size)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 체스보드 코너 검출
        found, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        if found:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # cv.drawChessboardCorners(frame, chessboard_size, corners_refined, found)
            
            # cv.solvePnP로 카메라 자세 (회전, 평행 이동) 추정
            ret_pnp, rvec, tvec = cv.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)
            if ret_pnp:
                # 전면과 후면 3D 점들을 영상 좌표계로 투영
                front_imgpts, _ = cv.projectPoints(front_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
                back_imgpts, _ = cv.projectPoints(back_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
                front_imgpts = front_imgpts.reshape(-1, 2).astype(int)
                back_imgpts = back_imgpts.reshape(-1, 2).astype(int)
                
                # 전면과 후면 외곽선을 그림
                cv.polylines(frame, [front_imgpts], isClosed=True, color=(0, 0, 255), thickness=2)
                cv.polylines(frame, [back_imgpts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # 전면과 후면의 대응 점들을 연결하여 측면(edge) 그리기
                n = len(front_imgpts)
                for i in range(n):
                    pt1 = tuple(front_imgpts[i])
                    pt2 = tuple(back_imgpts[i])
                    cv.line(frame, pt1, pt2, (255, 0, 0), 1)
        
        cv.imshow("3D Heart AR Overlay", frame)
        writer.write(frame) # 저장
        
        if cv.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    # 동영상 파일 경로 및 체스보드 설정
    video_file = 'chessboard.mp4'
    chessboard_size = (10, 7)
    square_size = 2.5
    target_size = None  # 필요에 따라 (width, height) 지정

    # 1. 캘리브레이션 데이터 수집 (검출된 프레임에서 체스보드 코너 추출)
    objpoints, imgpoints, image_size = detect_chessboard_corners(video_file, chessboard_size, square_size,
                                                                  display_delay=300, max_frames=20,
                                                                  frame_interval=0.5, target_size=target_size)
    if objpoints is None or len(objpoints) < 5:
        print("캘리브레이션에 필요한 충분한 체스보드 검출 결과가 없습니다!")
        exit(1)
    
    # 2. 캘리브레이션 수행
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, image_size)
    print("캘리브레이션 완료.")

    # 3. 하트 AR 오버레이 실행 (실시간으로 체스보드 평면에 하트 모양 오버레이)
    run_heart_ar(video_file, chessboard_size, square_size, camera_matrix, dist_coeffs, target_size=target_size)
