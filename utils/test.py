import matplotlib.pyplot as plt
import numpy as np

def cross_product(A, B, C):
    """A, B, C 세 점으로 구성된 벡터 AB와 AC의 외적 값을 계산한다."""
    AB = (B[0] - A[0], B[1] - A[1])
    AC = (C[0] - A[0], C[1] - A[1])
    return AB[0] * AC[1] - AB[1] * AC[0]

def is_left(A, B, C):
    """점 C가 선분 AB의 왼쪽(반시계방향)에 있는지 판별한다."""
    return cross_product(A, B, C) < 0

def point_in_region(point, line1, line2):
    """주어진 점이 어느 영역에 속하는지 판별한다.
    
    :param point: 확인하고자 하는 점 (x, y), 여기서 x, y는 0과 1 사이의 값
    :param line1: 첫 번째 선분을 이루는 두 점 [(x1, y1), (x2, y2)], 여기서 x1, y1, x2, y2는 0과 1 사이의 값
    :param line2: 두 번째 선분을 이루는 두 점 [(x3, y3), (x4, y4)], 여기서 x3, y3, x4, y4는 0과 1 사이의 값
    :return: 점이 속하는 영역 ('left', 'right', 'between')
    """
    left_of_line1 = is_left(line1[0], line1[1], point)
    left_of_line2 = is_left(line2[0], line2[1], point)

    print(left_of_line1)
    print(left_of_line2)

    
    if left_of_line1 and left_of_line2:
        return 'left'
    elif not left_of_line1 and left_of_line2:
        return 'center'
    else:
        return 'right'
    
def get_direction_by_dotproduct(x, y):
    point = (x, y)
    line1 = [(0.4, 1.0), (0.25, 0.0)]  # 첫 번째 선분
    line2 = [(0.6, 1.0), (0.75, 0.0)]  # 두 번째 선분

    # 영역 판별
    region = point_in_region(point, line1, line2)
    print(f"The point {point} is in the '{region}' region.")

    # 생성된 점을 검은색으로 표시
    if region == 'left':
        ret_str = 'diagonally to the left'
    elif region == 'right':
        ret_str = 'diagonally to the right'
    else:
        ret_str = 'in the center'

    return ret_str


# 이미지 크기 설정
plt.figure(figsize=(5, 5))

for i in range(1000):
    # 이미지 좌표계 내에서 랜덤한 점 생성
    x, y = np.random.rand(2)

    # 영역 판별
    ret_str = get_direction_by_dotproduct(x, y)
    print(ret_str)

    # 생성된 점을 검은색으로 표시
    if 'left' in ret_str:
        plt.scatter(x, y, color='red')
    elif 'right' in ret_str:
        plt.scatter(x, y, color='green')
    else:
        plt.scatter(x, y, color='blue')


# 축의 범위를 0에서 1로 설정
plt.xlim(0, 1)
plt.ylim(0, 1)

# 축 레이블 설정
plt.xlabel('X')
plt.ylabel('Y')

# 이미지 좌표계 표시를 위한 그리드 표시
plt.grid(True)

# 이미지 표시
plt.show()



