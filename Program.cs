using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

class Program
{
    public class ImageData
    {
        [ImageType(299, 299)] 
        public Bitmap Image { get; set; }
        public string ImagePath { get; set; }
    }

    public class ImagePrediction : ImageData
    {
        public float[] Score { get; set; } 
        public string PredictedLabel { get; set; } 
    }

    static void Main()
    {
        string modelPath = "Models/adv_inception_v3.onnx"; 
        var mlContext = new MLContext();

        var images = Directory.GetFiles("images"); 
        var imageData = new List<ImageData>();

        foreach (var imagePath in images)
        {
            imageData.Add(new ImageData { ImagePath = imagePath });
        }

        var imagePipeline = mlContext.Transforms.LoadImages("Image", "ImagePath")
            .Append(mlContext.Transforms.ResizeImages("Image", 299, 299))
            .Append(mlContext.Transforms.ExtractPixels("Image"))
            .Append(mlContext.Transforms.ApplyOnnxModel(modelPath));

        var imageDataView = mlContext.Data.LoadFromEnumerable(imageData);
        var transformer = imagePipeline.Fit(imageDataView);
        var predictions = transformer.Transform(imageDataView);

        var scoreColumn = predictions.GetColumn<float[]>("Score");
        foreach (var prediction in scoreColumn)
        {
            Console.WriteLine("Prediction Score: " + string.Join(",", prediction)); 
        }
    }
}
