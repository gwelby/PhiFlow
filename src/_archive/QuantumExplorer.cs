using System;
using System.Windows;
using System.Windows.Controls;
using System.Collections.ObjectModel;
using System.Windows.Markup;
using QuantumExplorer.Core;

namespace QuantumExplorer
{
    public class Explorer
    {
        private readonly QuantumSearch _search;
        private readonly QuantumGeometry _geometry;
        private readonly QuantumEffects _effects;
        private readonly QuantumPortals _portals;
        private readonly QuantumDimensions _dimensions;
        private readonly Canvas? _canvas;

        public Explorer(Canvas? canvas = null)
        {
            _canvas = canvas;
            _search = new QuantumSearch(_canvas);
            _geometry = new QuantumGeometry(_canvas);
            _effects = new QuantumEffects(_canvas);
            _portals = new QuantumPortals(_canvas);
            _dimensions = new QuantumDimensions(_canvas);

            InitializeQuantumComponents();
        }

        private void InitializeQuantumComponents()
        {
            // Initialize all components
            _geometry.Initialize();
            _effects.Initialize();
            _portals.Initialize();
            _dimensions.Initialize();
        }

        public void StartQuantumSystems()
        {
            if (_canvas == null) return;

            // Start all quantum systems
            _geometry.StartFlow();
            _effects.StartFlow();
            _portals.StartFlow();
            _dimensions.StartFlow();
        }

        public QuantumSearch Search => _search;
    }

    public class App : Application
    {
        [STAThread]
        public static void Main()
        {
            var app = new App();
            var mainWindow = new MainWindow();
            app.Run(mainWindow);
        }
    }

    public partial class MainWindow : Window
    {
        public ObservableCollection<QuantumFileInfo> fileCollection { get; set; }
        public ObservableCollection<QuantumTreeNode> treeNodes { get; set; }
        public Explorer _explorer { get; set; }
        public string CurrentPath { get; set; }

        public MainWindow()
        {
            fileCollection = new ObservableCollection<QuantumFileInfo>();
            treeNodes = new ObservableCollection<QuantumTreeNode>();
            CurrentPath = "D:\\";
            InitializeComponent();
            var canvas = (Canvas)FindName("QuantumCanvas");
            _explorer = new Explorer(canvas);
            DataContext = this;
            _explorer.StartQuantumSystems();
            SetupQuantumEvents();
        }

        private void SetupQuantumEvents()
        {
            if (HeartSearchButton != null) HeartSearchButton.Click += OnHeartSearch;
            if (VisionSearchButton != null) VisionSearchButton.Click += OnVisionSearch;
            if (QuantumSearchButton != null) QuantumSearchButton.Click += OnQuantumSearch;
            if (CrystalSearchButton != null) CrystalSearchButton.Click += OnCrystalSearch;
        }

        private async void OnHeartSearch(object sender, RoutedEventArgs e)
        {
            if (SearchBox == null) return;
            var results = await _explorer.Search.PerformHeartSearch(SearchBox.Text, CurrentPath);
            UpdateResults(results);
        }

        private async void OnVisionSearch(object sender, RoutedEventArgs e)
        {
            if (SearchBox == null) return;
            var results = await _explorer.Search.PerformVisionSearch(SearchBox.Text, CurrentPath);
            UpdateResults(results);
        }

        private async void OnQuantumSearch(object sender, RoutedEventArgs e)
        {
            if (SearchBox == null) return;
            var results = await _explorer.Search.PerformQuantumSearch(SearchBox.Text, CurrentPath);
            UpdateResults(results);
        }

        private async void OnCrystalSearch(object sender, RoutedEventArgs e)
        {
            if (SearchBox == null) return;
            var results = await _explorer.Search.PerformCrystalSearch(SearchBox.Text, CurrentPath);
            UpdateResults(results);
        }

        private void UpdateResults(List<QuantumFileInfo> results)
        {
            fileCollection.Clear();
            foreach (var result in results)
            {
                fileCollection.Add(result);
            }
        }
    }
}
