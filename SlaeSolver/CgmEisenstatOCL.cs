using Real = double;

using System.Diagnostics;

using SparkCL;
using OCLHelper;
using SparkAlgos.Types;

namespace SparkAlgos.SlaeSolver;

public class CgmEisenstatOCL : IDisposable, ISlaeSolver
{
    int _maxIter;
    Real _eps;

    int _n = 0; // размерность СЛАУ
    ComputeBuffer<Real> r_hat;
    ComputeBuffer<Real> r_stroke;
    ComputeBuffer<Real> p;
    ComputeBuffer<Real> t;
    ComputeBuffer<Real> Ap;
    ComputeBuffer<Real> z;
    
    ComputeBuffer<Real> dotpart;
    ComputeBuffer<Real> dotres;
    private bool disposedValue;

    public CgmEisenstatOCL(
        int maxIter,
        Real eps
    ) {
        _maxIter = maxIter;
        // TODO: уменьшение eps чтобы невязка ответа была сравнима с BicgStab
        _eps = eps / 1e+7;

        dotpart = new ComputeBuffer<Real>(32*2, BufferFlags.OnDevice);
        dotres  = new ComputeBuffer<Real>(1, BufferFlags.OnDevice);
    }

    public static ISlaeSolver Construct(int maxIter, double eps)
        => new CgmEisenstatOCL(maxIter, eps);

    // Выделить память для временных массивов
    // n - длина каждого массива
    public void AllocateTemps(int n)
    {
        if (n != _n)
        {
            _n = n;

            r_hat       = new (n, BufferFlags.OnDevice);
            r_stroke    = new (n, BufferFlags.OnDevice);
            p           = new (n, BufferFlags.OnDevice);
            t           = new (n, BufferFlags.OnDevice);
            Ap          = new (n, BufferFlags.OnDevice);
            z           = new (n, BufferFlags.OnDevice);
        }
    }

    public (Real discrep, int iter) Solve(IMatrix matrix, Span<Real> b, Span<Real> x)
    {
        if (matrix is IHalves m)
        {
            Console.WriteLine("Calling eisenstat");
            return SolveImpl(m, b, x);
        } else {
            throw new ArgumentException();
        }
    }
    
    static ComputeProgram? solvers;
    public (Real discrep, int iter) SolveImpl(IHalves matrix, Span<Real> b, Span<Real> x)
    {
        var sw = Stopwatch.StartNew();
        AllocateTemps(x.Length);

        var _x = new ComputeBuffer<Real>(x, BufferFlags.OnDevice);
        var _b = new ComputeBuffer<Real>(b, BufferFlags.OnDevice);

        // BiCGSTAB
        if (solvers == null)
        {
            solvers = new("SlaeSolver/Solvers.cl");
            Core.OnDeinit += () =>
            {
                solvers.Dispose();
                solvers = null;
            };
        }

        var kernVecMul = solvers.GetKernel(
            "VecMul",
            new NDRange((nuint)_x.Length/4).PadTo(Core.Prefered1D),
            new(Core.Prefered1D)
        );
        Event VecMulExecute(ComputeBuffer<Real> _res, ComputeBuffer<Real> _x) {
            kernVecMul.SetArg(0, _res);
            kernVecMul.SetArg(1, _x);
            kernVecMul.SetArg(2, _res.Length);
            return kernVecMul.Execute();
        }

        var sw_invL = new Stopwatch();
        var sw_invU = new Stopwatch();

        // TODO: set scratch buffers once per SBlas instance
        var SBlas = SparkAlgos.Blas.GetInstance();
        SBlas.Scratch64 = dotpart;
        SBlas.Scratch1 = dotres;

        Trace.WriteLine($"OpenCL prepare: {sw.ElapsedMilliseconds}ms");

        // 6a
        matrix.Mul(_x, r_stroke);
        _b.CopyDeviceTo(r_hat);
        SBlas.Axpy(-1, r_stroke, r_hat);
        matrix.InvLMul(r_hat);
        
        // 6b
        r_hat.CopyDeviceTo(r_stroke);
        VecMulExecute(r_stroke, matrix.Di);
        r_stroke.CopyDeviceTo(p);

        // precompute rr0
        var rr0 = SBlas.Dot(r_hat, r_stroke);
        
        int iter = 0;
        for (; iter < _maxIter; iter++)
        {
            // 6c
            // t:
            p.CopyDeviceTo(t);
            sw_invU.Start();
            matrix.InvUMul(t);
            sw_invU.Stop();
            // Ap:
            t.CopyDeviceTo(Ap);
            VecMulExecute(Ap, matrix.Di);
            SBlas.Scale(-1, Ap);
            SBlas.Axpy(1, p, Ap);
            sw_invL.Start();
            matrix.InvLMul(Ap);
            sw_invL.Stop();
            SBlas.Axpy(1, t, Ap);
            // alpha:
            var pAp = SBlas.Dot(p, Ap);
            var alpha = rr0 / pAp;
            
            // 6d
            SBlas.Axpy(alpha, t, _x);

            // 6e
            SBlas.Axpy(-alpha, Ap, r_hat);

            // 6g.1
            r_hat.CopyDeviceTo(r_stroke);
            VecMulExecute(r_stroke, matrix.Di);

            // 6g.2
            var rr1 = SBlas.Dot(r_hat, r_stroke);
            var b_hat = rr1 / rr0;

            // 6h
            SBlas.Scale(b_hat, p);
            SBlas.Axpy(1, r_stroke, p);

            rr0 = rr1;

            var rr = SBlas.Dot(r_hat, r_hat);
            var bb = SBlas.Dot(_b, _b);
            if (rr / bb < _eps)
            {
                break;
            }
        }

        Trace.WriteLine($"InvL time {sw_invL.ElapsedMilliseconds} ms");
        Trace.WriteLine($"InvU time {sw_invU.ElapsedMilliseconds} ms");

        matrix.Mul(_x, z);
        _b.CopyDeviceTo(r_hat);
        SBlas.Axpy(-1, z, r_hat);
        // BLAS.axpy(_x.Length, -1, t, r);
        var rr2 = SBlas.Dot(r_hat, r_hat);
        _x.DeviceReadTo(x);

        return (rr2, iter);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                // TODO: освободить управляемое состояние (управляемые объекты)
            }

            // TODO: надо вернуть обратно
            // r.Dispose();
            // r_hat.Dispose();
            // p.Dispose();
            // nu.Dispose();
            // h.Dispose();
            // s.Dispose();
            // t.Dispose();
            // dotpart.Dispose();
            // dotres.Dispose();
            // TODO: освободить неуправляемые ресурсы (неуправляемые объекты) и переопределить метод завершения
            // TODO: установить значение NULL для больших полей
            disposedValue = true;
        }
    }

    // // TODO: переопределить метод завершения, только если "Dispose(bool disposing)" содержит код для освобождения неуправляемых ресурсов
    ~CgmEisenstatOCL()
    {
        // Не изменяйте этот код. Разместите код очистки в методе "Dispose(bool disposing)".
        Dispose(disposing: false);
    }

    public void Dispose()
    {
        // Не изменяйте этот код. Разместите код очистки в методе "Dispose(bool disposing)".
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}
