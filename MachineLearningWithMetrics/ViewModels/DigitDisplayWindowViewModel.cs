using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace MachineLearningWithMetrics.ViewModels
{
    public class DigitDisplayWindowViewModel : Window
    {
        #region Fields

        public Grid PixelGrid { get; private set; }

        #endregion

        #region Constructor
        public DigitDisplayWindowViewModel(float[] pixels)
        {
            DrawDigits(pixels);
        }
        #endregion

        #region Private Methods
        private void DrawDigits(float[] pixels)
        {
            PixelGrid = new Grid();

            for (int i = 0; i < 28; i++)
            {
                PixelGrid.RowDefinitions.Add(new RowDefinition() { Height = new GridLength(10) });
                PixelGrid.ColumnDefinitions.Add(new ColumnDefinition() { Width = new GridLength(10) });
            }

            int pixelCount = 0;
            for (int i = 0; i < PixelGrid.RowDefinitions.Count; i++)
            {
                for (int j = 0; j < PixelGrid.ColumnDefinitions.Count; j++)
                {
                    Panel cellPanel = new DataGridCellsPanel();
                    byte rgbValue = (byte)pixels[pixelCount];
                    cellPanel.Background = new SolidColorBrush(Color.FromRgb(rgbValue, rgbValue, rgbValue));
                    Grid.SetRow(cellPanel, i);
                    Grid.SetColumn(cellPanel, j);
                    PixelGrid.Children.Add(cellPanel);
                    pixelCount++;
                }
            }

            this.Content = this.PixelGrid;

            this.SizeToContent = SizeToContent.WidthAndHeight;
        }
        #endregion
    }
}
