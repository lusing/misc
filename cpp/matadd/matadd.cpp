#include <mutex>
#include <thread>

// 矩阵维度
int width = 4;

// 矩阵
int A[width][width] = {/* 初始化 */};
int B[width][width] = {/* 初始化 */};
int C[width][width] = {0};

// 互斥锁
std::mutex mtx;

// 计算线程
void calculate(int row) {
  for (int col = 0; col < width; col++) {
    if (row < width && col < width) {
      mtx.lock();
      C[row][col] = A[row][col] + B[row][col];
      mtx.unlock();
    }
  }
}

int main() {
  // 创建线程
  std::thread t1(calculate, 0);
  std::thread t2(calculate, 1);
  std::thread t3(calculate, 2);
  std::thread t4(calculate, 3);

  // 等待线程结束
  t1.join();
  t2.join();
  t3.join();
  t4.join();

  // 打印结果
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      printf("%d ", C[i][j]);
    }
    printf("\n");
  }
}
