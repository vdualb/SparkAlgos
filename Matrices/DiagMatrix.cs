#if USE_DOUBLE
using Real = double;
#else
using Real = float;
#endif

using SparkCL;
using OCLHelper;

namespace SparkAlgos.Matrices;
using Types;

public ref struct DiagMatrixRef : IMatrixContainer
{
    // Left diagonal
    public required Span<Real> Ld3;
    public required Span<Real> Ld2;
    public required Span<Real> Ld1;
    public required Span<Real> Ld0;
    // Основная диагональ
    public required Span<Real> Di;
    // Right diagonal
    public required Span<Real> Rd0;
    public required Span<Real> Rd1;
    public required Span<Real> Rd2;
    public required Span<Real> Rd3;

    // Ld0 и Rd0 (*d0) находятся "вплотную" к основной диагонали
    // *d1, *d2, *d3 находятся стоят "вплотную" друг к другу
    // *d1 смещена на Gap элементов от *d0.
    // Например, если они находятся вплотную друг к друг,
    // то Gap == 1.
    public required int Gap;
    
    public int Size => Di.Length;
}

public class DiagMatrix : IHalves
{
#if USE_DOUBLE
    readonly string DefineUseDouble = "#define USE_DOUBLE";
#else
    readonly string DefineUseDouble = string.Empty;
#endif
    // Left diagonal
    public ComputeBuffer<Real> Ld3;
    public ComputeBuffer<Real> Ld2;
    public ComputeBuffer<Real> Ld1;
    public ComputeBuffer<Real> Ld0;
    // Основная диагональ
    public ComputeBuffer<Real> Di;
    // Right diagonal
    public ComputeBuffer<Real> Rd0;
    public ComputeBuffer<Real> Rd1;
    public ComputeBuffer<Real> Rd2;
    public ComputeBuffer<Real> Rd3;

    // Ld0 и Rd0 (*d0) находятся "вплотную" к основной диагонали
    // *d1, *d2, *d3 находятся стоят "вплотную" друг к другу
    // *d1 смещена на Gap элементов от *d0.
    // Например, если они находятся вплотную друг к друг,
    // то Gap == 1.
    public int Gap;

    public int Size => Di.Length;
    ComputeBuffer<Real> IMatrix.Di => Di;

    static SparkCL.Kernel? kernMul;
    static SparkCL.Kernel? kernLMul;
    static SparkCL.Kernel? kernUMul;
    static SparkCL.Kernel? kernInvLMul;
    static SparkCL.Kernel? kernInvUMul;

    public DiagMatrix(DiagMatrixRef matrix)
    {
        Ld3 = new ComputeBuffer<Real>(matrix.Ld3, BufferFlags.OnDevice);
        Ld2 = new ComputeBuffer<Real>(matrix.Ld2, BufferFlags.OnDevice);
        Ld1 = new ComputeBuffer<Real>(matrix.Ld1, BufferFlags.OnDevice);
        Ld0 = new ComputeBuffer<Real>(matrix.Ld0, BufferFlags.OnDevice);
        Di = new ComputeBuffer<Real>(matrix.Di, BufferFlags.OnDevice);
        Rd0 = new ComputeBuffer<Real>(matrix.Rd0, BufferFlags.OnDevice);
        Rd1 = new ComputeBuffer<Real>(matrix.Rd1, BufferFlags.OnDevice);
        Rd2 = new ComputeBuffer<Real>(matrix.Rd2, BufferFlags.OnDevice);
        Rd3 = new ComputeBuffer<Real>(matrix.Rd3, BufferFlags.OnDevice);
    
        Gap = matrix.Gap;
    }

    public void Mul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res)
    {
        if (kernMul == null)
        {
            var support = ComputeProgram.FromFilename("Matrices/DiagMatrix.cl", DefineUseDouble);
            var localWork = new NDRange(Core.Prefered1D);

            kernMul = support.GetKernel(
                "DiagMul",
                new NDRange((nuint)vec.Length).PadTo(Core.Prefered1D),
                localWork
            );
            Core.OnDeinit += () =>
            {
                kernMul.Dispose();
                kernMul = null;
            };
        }
            kernMul.GlobalWork = new NDRange((nuint)vec.Length).PadTo(Core.Prefered1D);
            kernMul.SetArg(0, Ld3);
            kernMul.SetArg(1, Ld2);
            kernMul.SetArg(2, Ld1);
            kernMul.SetArg(3, Ld0);
            kernMul.SetArg(4, Di);
            kernMul.SetArg(5, Rd0);
            kernMul.SetArg(6, Rd1);
            kernMul.SetArg(7, Rd2);
            kernMul.SetArg(8, Rd3);
            kernMul.SetArg(9, vec.Length);
            kernMul.SetArg(10, Gap);

        kernMul.SetArg(11, vec);
        kernMul.SetArg(12, res);

        kernMul.Execute();
    }
    
    public void LMul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res)
    {
        if (kernLMul == null)
        {
            var support = ComputeProgram.FromFilename("Matrices/DiagMatrix.cl", DefineUseDouble);
            var localWork = new NDRange(Core.Prefered1D);

            kernLMul = support.GetKernel(
                "LMul",
                new NDRange((nuint)vec.Length).PadTo(Core.Prefered1D),
                localWork
            );
            Core.OnDeinit += () =>
            {
                kernLMul.Dispose();
                kernLMul = null;
            };
        }
        
        kernLMul.GlobalWork = new NDRange((nuint)vec.Length).PadTo(Core.Prefered1D);
        kernLMul.SetArg(0, Ld3);
        kernLMul.SetArg(1, Ld2);
        kernLMul.SetArg(2, Ld1);
        kernLMul.SetArg(3, Ld0);
        
        kernLMul.SetArg(4, Di);
        
        kernLMul.SetArg(5, vec.Length);
        kernLMul.SetArg(6, Gap);

        kernLMul.SetArg(7, vec);
        kernLMul.SetArg(8, res);
    
        kernLMul.Execute();
    }

    public void UMul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res)
    {
        if (kernUMul == null)
        {
            var support = ComputeProgram.FromFilename("Matrices/DiagMatrix.cl", DefineUseDouble);
            var localWork = new NDRange(Core.Prefered1D);

            kernUMul = support.GetKernel(
                "UMul",
                new NDRange((nuint)vec.Length).PadTo(Core.Prefered1D),
                localWork
            );
            Core.OnDeinit += () =>
            {
                kernUMul.Dispose();
                kernUMul = null;
            };
        }
        
        kernUMul.GlobalWork = new NDRange((nuint)vec.Length).PadTo(Core.Prefered1D);
        kernUMul.SetArg(0, Rd3);
        kernUMul.SetArg(1, Rd2);
        kernUMul.SetArg(2, Rd1);
        kernUMul.SetArg(3, Rd0);
        
        kernUMul.SetArg(4, Di);
        
        kernUMul.SetArg(5, vec.Length);
        kernUMul.SetArg(6, Gap);

        kernUMul.SetArg(7, vec);
        kernUMul.SetArg(8, res);
    
        kernUMul.Execute();
    }

    public void InvLMul(ComputeBuffer<Real> inOut)
    {
        if (kernInvLMul == null)
        {
            var support = ComputeProgram.FromFilename("Matrices/DiagMatrix.cl", DefineUseDouble);
            var globalWork = new NDRange(4);
            var localWork = new NDRange(4);

            kernInvLMul = support.GetKernel(
                "InvLMul",
                globalWork,
                localWork
            );
            Core.OnDeinit += () =>
            {
                kernInvLMul.Dispose();
                kernInvLMul = null;
            };
        }
        
        kernInvLMul.SetArg(0, Ld3);
        kernInvLMul.SetArg(1, Ld2);
        kernInvLMul.SetArg(2, Ld1);
        kernInvLMul.SetArg(3, Ld0);
        
        kernInvLMul.SetArg(4, Di);
        
        kernInvLMul.SetArg(5, inOut.Length);
        kernInvLMul.SetArg(6, Gap);

        kernInvLMul.SetArg(7, inOut);
    
        kernInvLMul.Execute();
    }

    public void InvUMul(ComputeBuffer<Real> inOut)
    {
        if (kernInvUMul == null)
        {
            var support = ComputeProgram.FromFilename("Matrices/DiagMatrix.cl", DefineUseDouble);
            var globalWork = new NDRange(4);
            var localWork = new NDRange(4);

            kernInvUMul = support.GetKernel(
                "InvUMul",
                globalWork,
                localWork
            );
            Core.OnDeinit += () =>
            {
                kernInvUMul.Dispose();
                kernInvUMul = null;
            };
        }
        
        kernInvUMul.SetArg(0, Rd3);
        kernInvUMul.SetArg(1, Rd2);
        kernInvUMul.SetArg(2, Rd1);
        kernInvUMul.SetArg(3, Rd0);
        
        kernInvUMul.SetArg(4, Di);
        
        kernInvUMul.SetArg(5, inOut.Length);
        kernInvUMul.SetArg(6, Gap);

        kernInvUMul.SetArg(7, inOut);
    
        kernInvUMul.Execute();
    }
}
