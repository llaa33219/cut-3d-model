import trimesh
import pyvista as pv
import numpy as np
import os
from PIL import Image

def get_face_color(face_idx, mesh, texture_image):
    """
    주어진 얼굴의 UV 좌표를 사용하여 텍스처 이미지에서 평균 색상을 샘플링합니다.

    Parameters:
        face_idx (int): 얼굴 인덱스.
        mesh (trimesh.Trimesh): 삼각형 메쉬 객체.
        texture_image (PIL.Image): 텍스처 이미지 객체.

    Returns:
        tuple: (R, G, B) 형식의 색상 튜플.
    """
    # 얼굴의 정점 인덱스 가져오기
    face = mesh.faces[face_idx]
    # 각 정점의 UV 좌표 가져오기
    uv = mesh.visual.uv[face]  # (3, 2)

    # 텍스처 이미지 크기
    tex_width, tex_height = texture_image.size

    colors = []
    for coord in uv:
        u, v = coord
        # UV 좌표를 이미지 픽셀 좌표로 변환 (v는 이미지의 상단이 0)
        x = int(u * (tex_width - 1))
        y = int((1 - v) * (tex_height - 1))
        # 색상 샘플링
        color = texture_image.getpixel((x, y))
        colors.append(color[:3])  # RGB만 사용

    # 평균 색상 계산
    avg_color = tuple(np.mean(colors, axis=0).astype(int))
    return avg_color

# 1. 3D 모델 파일 경로 설정 (OBJ 파일로 변경)
model_path = 'damas mk2.obj'  # 실제 OBJ 파일 경로로 변경하세요

# 2. Trimesh를 사용하여 OBJ 파일 로드
try:
    mesh_trimesh = trimesh.load(model_path, force='mesh')
    print(f"모델 '{model_path}' 로드 완료.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit(1)

# 3. Scene인지 Mesh인지 확인하고 처리
if isinstance(mesh_trimesh, trimesh.Scene):
    print("씬(Scene) 객체가 로드되었습니다. 모든 메쉬를 하나로 병합합니다.")
    try:
        # 씬 내의 모든 메쉬를 하나로 병합
        combined_mesh = trimesh.util.concatenate(mesh_trimesh.dump())
        print("메쉬 병합 완료.")
    except Exception as e:
        print(f"메쉬 병합 중 오류 발생: {e}")
        exit(1)
elif isinstance(mesh_trimesh, trimesh.Trimesh):
    combined_mesh = mesh_trimesh
    print("단일 메쉬 객체가 로드되었습니다.")
else:
    raise TypeError("Trimesh가 지원하지 않는 메쉬 타입을 로드했습니다.")

# 4. 메쉬 삼각분할 (Triangulation)
if combined_mesh.faces.shape[1] != 3:
    print("메쉬를 삼각분할합니다.")
    combined_mesh = combined_mesh.triangulate()
    print("삼각분할 완료.")
else:
    print("메쉬는 이미 삼각형으로 분할되어 있습니다.")

# 4-1. 메쉬 회전 (90도)
# 회전 축과 각도를 설정합니다. 여기서는 x축을 기준으로 90도 회전합니다.
rotation_axis = 'x'  # 'x', 'y', 'z' 중 선택
rotation_angle = 90  # 회전 각도 (도 단위)

# 회전 행렬 생성
if rotation_axis.lower() == 'x':
    rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(rotation_angle), [1, 0, 0]
    )
elif rotation_axis.lower() == 'y':
    rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(rotation_angle), [0, 1, 0]
    )
elif rotation_axis.lower() == 'z':
    rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(rotation_angle), [0, 0, 1]
    )
else:
    raise ValueError("회전 축은 'x', 'y', 또는 'z' 중 하나여야 합니다.")

# 메쉬에 회전 적용
combined_mesh.apply_transform(rotation_matrix)
print(f"메쉬를 {rotation_axis.upper()} 축을 기준으로 {rotation_angle}도 회전했습니다.")

# 5. 텍스처 이미지 로드
if combined_mesh.visual.kind == 'texture':
    print("텍스처가 있는 메쉬입니다.")
    # 텍스처 이미지 가져오기
    texture_image_obj = combined_mesh.visual.material.image
    if texture_image_obj is None:
        print("텍스처 이미지가 로드되지 않았습니다.")
        exit(1)
    try:
        if isinstance(texture_image_obj, Image.Image):
            texture_image = texture_image_obj.convert("RGB")
            print("텍스처 이미지가 PIL Image 객체로 로드되었습니다.")
        else:
            # 다른 타입인 경우 (예: 파일 경로), Image.open 사용
            texture_image = Image.open(texture_image_obj).convert("RGB")
            print("텍스처 이미지 로드 완료.")
    except Exception as e:
        print(f"텍스처 이미지 로드 중 오류 발생: {e}")
        exit(1)
    
    # UV 좌표 가져오기
    if hasattr(combined_mesh.visual, 'uv') and combined_mesh.visual.uv is not None:
        uv_coords = combined_mesh.visual.uv  # (n_vertices, 2)
        print("UV 좌표 추출 완료.")
    else:
        print("UV 좌표가 없습니다.")
        exit(1)
    
    # 각 얼굴의 색상 계산
    face_colors = []
    for face_idx in range(combined_mesh.faces.shape[0]):
        color = get_face_color(face_idx, combined_mesh, texture_image)
        face_colors.append(color)
    face_colors = np.array(face_colors)
    print("각 얼굴의 색상 계산 완료.")
else:
    print("텍스처가 없는 메쉬입니다. 기본 색상을 사용합니다.")
    # 기본 색상 (예: 회색)
    face_colors = np.tile([128, 128, 128], (combined_mesh.faces.shape[0], 1))

# 6. Trimesh Mesh를 PyVista Mesh로 변환
try:
    # Trimesh의 vertices는 (n, 3), faces는 (m, 3) 형태입니다.
    # PyVista는 faces 배열을 (n_faces * 4) 형태로 필요로 합니다. (첫 번째 값은 각 면의 정점 수)
    faces_pv = np.hstack([np.full((combined_mesh.faces.shape[0], 1), 3), combined_mesh.faces]).flatten()
    mesh_pv = pv.PolyData(combined_mesh.vertices, faces_pv)
    print("Trimesh 메쉬를 PyVista PolyData로 변환 완료.")
except Exception as e:
    print(f"Trimesh에서 PyVista로 변환 중 오류 발생: {e}")
    exit(1)

# 7. 얼굴 색상 데이터 추가
# PyVista는 cell data를 지원하므로, cell_data에 face_colors를 추가합니다.
try:
    mesh_pv.cell_data['Colors'] = face_colors
    print("얼굴 색상 데이터가 cell_data에 추가되었습니다.")
except AttributeError:
    try:
        # PyVista 버전이 낮아 cell_data가 없을 경우
        mesh_pv['Colors'] = face_colors
        print("얼굴 색상 데이터가 직접 추가되었습니다.")
    except Exception as e:
        print(f"얼굴 색상 데이터 추가 중 오류 발생: {e}")
        exit(1)

# 8. 셀 데이터를 포인트 데이터로 변환하여 포인트 색상 추가
mesh_pv = mesh_pv.cell_data_to_point_data('Colors')
print("cell_data 'Colors'를 point_data로 변환 완료.")

# 9. 모델의 경계 상자 정보 확인
bounds = mesh_pv.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
z_min, z_max = bounds[4], bounds[5]
print(f"모델 Z 범위: {z_min} ~ {z_max}")

# 10. 슬라이스 개수 설정 (360개로 변경)
num_slices = 360  # 슬라이스 개수를 360으로 설정

# 11. 슬라이스 간격 계산
z_values = np.linspace(z_min, z_max, num_slices + 1)

# 12. 저장할 이미지 폴더 생성
output_dir = 'slices_images'
os.makedirs(output_dir, exist_ok=True)
print(f"이미지 저장 폴더 '{output_dir}' 생성 또는 존재 확인.")

# 13. 슬라이스 생성 및 이미지 저장
for i in range(num_slices):
    # 현재 슬라이스의 높이
    z = (z_values[i] + z_values[i + 1]) / 2

    # 슬라이스 수행
    slice_pv = mesh_pv.slice(normal=(0, 0, 1), origin=(0, 0, z))

    # 단면이 존재하는 경우에만 처리
    if slice_pv.n_points > 0:
        try:
            # 슬라이스의 포인트 색상 가져오기
            if 'Colors' in slice_pv.point_data:
                slice_colors = slice_pv.point_data['Colors']
                if slice_colors.size > 0:
                    # 색상을 [0, 1] 범위로 정규화
                    colors_normalized = slice_colors / 255.0
                else:
                    # 기본 색상 설정 (예: 빨간색)
                    colors_normalized = np.array([[1, 0, 0]])
            else:
                # 기본 색상 설정 (예: 빨간색)
                colors_normalized = np.array([[1, 0, 0]])

            # 플롯터 생성
            plotter = pv.Plotter(off_screen=True)  # off_screen=True로 백그라운드 렌더링

            # 배경을 투명하게 설정
            plotter.set_background((0, 0, 0, 0))  # 완전 투명

            # 슬라이스 선만 추가 (포인트 색상 사용)
            plotter.add_mesh(slice_pv, scalars='Colors', rgb=True, line_width=2, style='wireframe', render_lines_as_tubes=True)

            # 카메라 설정 (위에서 바라보기)
            plotter.camera_position = 'xy'  # 'xy'는 위에서 아래로 바라보는 시점

            # 렌더링 크기 설정 (원하는 해상도로 조정 가능)
            plotter.window_size = [800, 800]

            # 이미지 렌더링 및 저장 (transparent_background=True로 투명 배경 설정)
            screenshot_path = os.path.join(output_dir, f'slice_{i:03d}.png')
            plotter.screenshot(screenshot_path, transparent_background=True)
            plotter.close()

            print(f"'{screenshot_path}' 저장 완료.")
        except Exception as e:
            print(f"슬라이스 {i}에서 이미지 저장 중 오류 발생: {e}")
    else:
        print(f"z={z:.2f} 위치에 단면이 존재하지 않습니다.")
