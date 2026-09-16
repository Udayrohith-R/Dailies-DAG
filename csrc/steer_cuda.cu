// CUDA stub for SFS-style steer-add: h' = h + alpha * v (elementwise).
// Full fused kernel lives in csrc/steer_triton.py (Triton). This file documents
// the intended launch geometry for a hand-written CUDA port.
__global__ void steer_add_kernel(const float* h, const float* v, float* out,
                                 int n, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = h[i] + alpha * v[i];
}
