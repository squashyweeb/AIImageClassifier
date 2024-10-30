using System;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors; // Required for tensor operations
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ImageRecognitionApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Check if the correct number of arguments is provided
            if (args.Length < 2)
            {
                Console.WriteLine("Usage: ImageRecognitionApp <model_path> <image_path>");
                return;
            }

            string modelPath = args[0];
            string imagePath = args[1];

            // Load model and predict
            LoadAndPredict(modelPath, imagePath);
        }

        static void LoadAndPredict(string modelPath, string imagePath)
        {
            // Validate model file existence
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Model file not found: {modelPath}");
                return;
            }

            // Validate image file existence
            if (!File.Exists(imagePath))
            {
                Console.WriteLine($"Image file not found: {imagePath}");
                return;
            }

            // Load ONNX model
            using var session = new InferenceSession(modelPath);

            // Load image using ImageSharp
            using (Image<Rgba32> image = Image.Load<Rgba32>(imagePath))
            {
                // Resize image to model input size (e.g., 299x299)
                image.Mutate(x => x.Resize(299, 299));

                // Prepare input data for the model
                var inputData = new float[1 * 3 * 299 * 299];
                for (int y = 0; y < 299; y++)
                {
                    for (int x = 0; x < 299; x++)
                    {
                        var pixel = image[x, y];
                        inputData[0 * 3 * 299 * 299 + 0 * 299 * 299 + y * 299 + x] = pixel.R / 255f; // Red
                        inputData[0 * 3 * 299 * 299 + 1 * 299 * 299 + y * 299 + x] = pixel.G / 255f; // Green
                        inputData[0 * 3 * 299 * 299 + 2 * 299 * 299 + y * 299 + x] = pixel.B / 255f; // Blue
                    }
                }

                // Create input tensor
                var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 3, 299, 299 }); // DenseTensor for input
                var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) }; // Replace "input" with actual input name

                // Run inference
                using var results = session.Run(inputs);
                var outputTensor = results.First().AsTensor<float>();

                // Get predicted class index by finding max value
                var maxIndex = 0;
                var maxValue = float.MinValue;

                for (int i = 0; i < outputTensor.Length; i++)
                {
                    if (outputTensor[i] > maxValue)
                    {
                        maxValue = outputTensor[i];
                        maxIndex = i;
                    }
                }

                // Output predicted class index
                Console.WriteLine($"Predicted class index: {maxIndex}");
            }
        }
    }
}
