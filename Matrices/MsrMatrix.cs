using SparkCL;
using OCLHelper;

namespace SparkAlgos.Matrices;
using Types;

public ref struct MsrMatrixRef : IMatrixContainer
{
    public required Span<Real> Elems;
    public required Span<Real> Di;
    public required Span<int> Ia;
    public required Span<int> Ja;

    public int Size => Di.Length;
}

public class MsrMatrix : IHalves
{
#if USE_DOUBLE
    readonly string DefineUseDouble = "#define USE_DOUBLE";
#else
    readonly string DefineUseDouble = string.Empty;
#endif
    public ComputeBuffer<Real> Elems;
    public ComputeBuffer<Real> Di;
    public ComputeBuffer<int> Ia;
    public ComputeBuffer<int> Ja;

    public int Size => Di.Length;
    ComputeBuffer<Real> IMatrix.Di => Di;

    static SparkCL.Kernel? kernMul;
    static SparkCL.Kernel? kernLMul;
    static SparkCL.Kernel? kernUMul;
    static SparkCL.Kernel? kernInvLMul;
    static SparkCL.Kernel? kernInvUMul;
    
    public MsrMatrix(MsrMatrixRef matrix)
    {
        Elems = new ComputeBuffer<Real> (matrix.Elems, BufferFlags.OnDevice);
        Ia    = new ComputeBuffer<int>  (matrix.Ia, BufferFlags.OnDevice);
        Ja    = new ComputeBuffer<int>  (matrix.Ja, BufferFlags.OnDevice);
        Di    = new ComputeBuffer<Real> (matrix.Di, BufferFlags.OnDevice);
    }

    public void Mul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res)
    {
        if (kernMul == null)
        {
            var support = ComputeProgram.FromFilename("Matrices/MsrMatrix.cl", DefineUseDouble);
            var localWork = new NDRange(Core.Prefered1D);

            kernMul = support.GetKernel(
                "MsrMul",
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
            kernMul.SetArg(0, Elems);
            kernMul.SetArg(1, Di);
            kernMul.SetArg(2, Ia);
            kernMul.SetArg(3, Ja);
            kernMul.SetArg(4, vec.Length);

        kernMul.SetArg(5, vec);
        kernMul.SetArg(6, res);

        kernMul.Execute();
    }
    
    
    public void LMul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res)
    {
        if (kernLMul == null)
        {
            var support = ComputeProgram.FromFilename("Matrices/MsrMatrix.cl", DefineUseDouble);
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
            kernLMul.SetArg(0, Elems);
            kernLMul.SetArg(1, Di);
            kernLMul.SetArg(2, Ia);
            kernLMul.SetArg(3, Ja);
            kernLMul.SetArg(4, vec.Length);
    
        kernLMul.SetArg(5, vec);
        kernLMul.SetArg(6, res);
    
        kernLMul.Execute();
    }

    public void UMul(ComputeBuffer<Real> vec, ComputeBuffer<Real> res)
    {
        if (kernUMul == null)
        {
            var support = ComputeProgram.FromFilename("Matrices/MsrMatrix.cl", DefineUseDouble);
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
            kernUMul.SetArg(0, Elems);
            kernUMul.SetArg(1, Di);
            kernUMul.SetArg(2, Ia);
            kernUMul.SetArg(3, Ja);
            kernUMul.SetArg(4, vec.Length);
    
        kernUMul.SetArg(5, vec);
        kernUMul.SetArg(6, res);
    
        kernUMul.Execute();
    }

    public void InvLMul(ComputeBuffer<Real> inOut)
    {
        if (kernInvLMul == null)
        {
            var support = ComputeProgram.FromFilename("Matrices/MsrMatrix.cl", DefineUseDouble);
            var globalWork = new NDRange(128);
            var localWork = new NDRange(128);

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
            kernInvLMul.SetArg(0, Elems);
            kernInvLMul.SetArg(1, Di);
            kernInvLMul.SetArg(2, Ia);
            kernInvLMul.SetArg(3, Ja);
            kernInvLMul.SetArg(4, inOut.Length);
    
        kernInvLMul.SetArg(5, inOut);
    
        kernInvLMul.Execute();
    }

    public void InvUMul(ComputeBuffer<Real> inOut)
    {
        if (kernInvUMul == null)
        {
            var support = ComputeProgram.FromFilename("Matrices/MsrMatrix.cl", DefineUseDouble);
            var globalWork = new NDRange(128);
            var localWork = new NDRange(128);

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
            kernInvUMul.SetArg(0, Elems);
            kernInvUMul.SetArg(1, Di);
            kernInvUMul.SetArg(2, Ia);
            kernInvUMul.SetArg(3, Ja);
            kernInvUMul.SetArg(4, inOut.Length);
    
        kernInvUMul.SetArg(5, inOut);
    
        kernInvUMul.Execute();
    }
}
