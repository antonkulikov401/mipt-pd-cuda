#include <iostream>
#include <algorithm>
#include <cmath>
#include "matrix.hpp"

size_t IMatrix::width() const {
  return _width;
}

size_t IMatrix::height() const {
  return _height;
}

size_t IMatrix::size() const {
  return _width * _height;
}

float* IMatrix::data() {
  return _data;
}

const float* IMatrix::data() const {
  return _data;
}

float& IMatrix::operator()(size_t i, size_t j) {
  return _data[_width * i + j];
}

float IMatrix::operator()(size_t i, size_t j) const {
  return _data[_width * i + j];
}

IMatrix::~IMatrix() {}

CPU_Matrix::CPU_Matrix(size_t height, size_t width) : IMatrix(height, width) {
  _data = new float[size()];
}

CPU_Matrix::CPU_Matrix(const CPU_Matrix& other) 
  : CPU_Matrix(other._height, other._width) {
  cudaMemcpy(_data, other._data, size() * sizeof(float), cudaMemcpyHostToHost);
}

CPU_Matrix::CPU_Matrix(const GPU_Matrix& other) 
  : CPU_Matrix(other.height(), other.width()) {
  cudaMemcpy(_data, other.data(), size() * sizeof(float), cudaMemcpyDeviceToHost);
}

bool CPU_Matrix::operator==(const CPU_Matrix& other) {
  if (_width != other._width || _height != other._height) {
    return false;
  }
  return std::equal(_data, _data + size(), other._data);
}

CPU_Matrix::~CPU_Matrix() {
  delete[] _data;
}

void CPU_Matrix::print() const {
  for (size_t row = 0; row < _height; ++row) {
    for (size_t col = 0; col < _width; ++col) {
      std::cout << operator()(row, col) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

GPU_Matrix::GPU_Matrix(size_t height, size_t width) : IMatrix(height, width) {
  cudaMallocManaged(&_data, size() * sizeof(float));
}

GPU_Matrix::GPU_Matrix(const GPU_Matrix& other)
  : GPU_Matrix(other._height, other._width) {
  cudaMemcpy(_data, other._data, size() * sizeof(float), cudaMemcpyDeviceToDevice);
}

GPU_Matrix::GPU_Matrix(const CPU_Matrix& other) 
  : GPU_Matrix(other.height(), other.width()) {
  cudaMemcpy(_data, other.data(), size() * sizeof(float), cudaMemcpyHostToDevice);
}

GPU_Matrix::~GPU_Matrix() {
  cudaFree(_data);
}
