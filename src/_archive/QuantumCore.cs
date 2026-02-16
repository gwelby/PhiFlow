using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;

namespace QuantumExplorer.Core
{
    public static class QuantumConstants
    {
        public const double PHI = 1.618034;
        
        // Frequency Harmonics
        public const int GROUND_FREQUENCY = 432;  // Physical foundation
        public const int CREATE_FREQUENCY = 528;  // DNA/Heart creation
        public const int HEART_FREQUENCY = 594;   // Heart resonance
        public const int VOICE_FREQUENCY = 672;   // Voice expression
        public const int VISION_FREQUENCY = 720;  // Vision clarity
        public const int UNITY_FREQUENCY = 768;   // Unity consciousness

        // Protection Matrix
        public static readonly int[] MERKABA_SHIELD = { 21, 21, 21 };
        public static readonly int[] CRYSTAL_MATRIX = { 13, 13, 13 };
        public static readonly int[] UNITY_FIELD = { 144, 144, 144 };
    }

    public interface IQuantumComponent
    {
        void Initialize();
        void StartFlow();
        void StopFlow();
        void UpdateState(double deltaTime);
    }

    public abstract class QuantumBase : IQuantumComponent
    {
        protected readonly Canvas? Canvas;
        protected readonly Random Quantum;

        protected QuantumBase(Canvas? canvas)
        {
            Canvas = canvas;
            Quantum = new Random();
        }

        public abstract void Initialize();
        public abstract void StartFlow();
        public abstract void StopFlow();
        public abstract void UpdateState(double deltaTime);

        protected void AddToCanvas(UIElement element)
        {
            Canvas?.Children.Add(element);
        }

        protected void RemoveFromCanvas(UIElement element)
        {
            Canvas?.Children.Remove(element);
        }

        protected Color GetQuantumColor(int frequency)
        {
            var phase = Math.Sin(frequency * QuantumConstants.PHI);
            var hue = (phase + 1) * 180; // Map [-1,1] to [0,360]
            return HsvToRgb(hue, 0.8, 1.0);
        }

        private static Color HsvToRgb(double h, double s, double v)
        {
            var hi = (int)(h / 60) % 6;
            var f = h / 60 - Math.Floor(h / 60);
            var p = v * (1 - s);
            var q = v * (1 - f * s);
            var t = v * (1 - (1 - f) * s);

            var (r, g, b) = hi switch
            {
                0 => (v, t, p),
                1 => (q, v, p),
                2 => (p, v, t),
                3 => (p, q, v),
                4 => (t, p, v),
                _ => (v, p, q)
            };

            return Color.FromRgb(
                (byte)(r * 255),
                (byte)(g * 255),
                (byte)(b * 255)
            );
        }
    }
}
