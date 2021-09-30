//
// Rectangular real-valued matrix interface
//
class IMatrix {
public:
  IMatrix(size_t height, size_t width) : _width(width), _height(height) {}
  virtual ~IMatrix() = 0;

  size_t width() const;
  size_t height() const;
  size_t size() const;
  float* data();
  const float* data() const;
  float& operator()(size_t i, size_t j);
  float operator()(size_t i, size_t j) const;

protected:
  size_t _height;
  size_t _width;
  float* _data;
};

class GPU_Matrix;

class CPU_Matrix : public IMatrix {
public:
  template<typename GenT>
  CPU_Matrix(size_t height, size_t width, GenT generator);
  CPU_Matrix(size_t height, size_t width);
  CPU_Matrix(const CPU_Matrix& other);
  CPU_Matrix(const GPU_Matrix& other);
  CPU_Matrix(CPU_Matrix&& other) = default;
  CPU_Matrix& operator=(const CPU_Matrix& other) = delete;
  CPU_Matrix& operator=(CPU_Matrix&& other) = delete;
  ~CPU_Matrix();
  bool operator==(const CPU_Matrix& other);

  void print() const;
};

class GPU_Matrix : public IMatrix {
public:
  GPU_Matrix(size_t height, size_t width);
  GPU_Matrix(const GPU_Matrix& other);
  GPU_Matrix(const CPU_Matrix& other);
  GPU_Matrix(GPU_Matrix&& other) = default;
  GPU_Matrix& operator=(const GPU_Matrix& other) = delete;
  GPU_Matrix& operator=(GPU_Matrix&& other) = delete;
  ~GPU_Matrix();
};

template<typename GeneratorT>
CPU_Matrix::CPU_Matrix(size_t height, size_t width, GeneratorT generator) 
  : CPU_Matrix(height, width) {
  for (size_t i = 0; i < size(); ++i) {
    _data[i] = generator();
  }
}
