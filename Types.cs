using SparkCL;

namespace SparkAlgos.Types;

public interface IMatrixContainer
{
    int Size { get; }
};

public interface IMatrix
{
    int Size { get; }
    // TODO: нужно для предобуславливания.
    // Надо придумать что-то более разумное
    ComputeBuffer<Real> Di { get; }

    void Mul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res);
}

public interface IHalves : IMatrix
{
    /// нижний треугольник на вектор
    void LMul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res);
    /// верхний треугольник на вектор
    void UMul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res);
    
    /// in-place решение L*x=f для x
    void InvLMul(ComputeBuffer<Real> inOut);
    /// in-place решение U*x=f для x
    void InvUMul(ComputeBuffer<Real> inOut);
}
