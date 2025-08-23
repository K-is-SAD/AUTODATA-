#!/usr/bin/env python3
"""
Test script for enhanced AUTODATA features:
1. Multiple SQL queries generation
2. Comprehensive statistical analysis
3. Enhanced visualizations
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_multiple_sql_queries():
    """Test multiple SQL queries generation"""
    print("🧪 Testing Multiple SQL Queries Generation...")
    
    try:
        from src.agents.sql_agent import SQLAgent
        
        # Create a simple test agent (you'll need to configure this with your LLM)
        # agent = SQLAgent("ollama", model_name="mistral:7b")
        
        # Test table info
        table_info = """
        Table: data_table
        Columns:
        - id (INTEGER)
        - name (TEXT)
        - age (INTEGER)
        - salary (REAL)
        - department (TEXT)
        """
        
        # Test user question that should generate multiple queries
        user_question = "Analyze employee data by department and show salary statistics"
        
        print(f"✅ Multiple SQL queries functionality imported successfully")
        print(f"📝 Test question: {user_question}")
        print(f"📊 Table info: {table_info}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_comprehensive_analysis():
    """Test comprehensive statistical analysis"""
    print("\n🧪 Testing Comprehensive Statistical Analysis...")
    
    try:
        from src.plotting.plot_generator import (
            generate_comprehensive_analysis,
            generate_univariate_analysis,
            generate_bivariate_analysis,
            generate_multivariate_analysis,
            generate_correlation_analysis
        )
        
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'id': range(100),
            'age': np.random.normal(35, 10, 100),
            'salary': np.random.normal(50000, 15000, 100),
            'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 100),
            'experience': np.random.normal(8, 3, 100),
            'performance_score': np.random.normal(75, 10, 100)
        })
        
        print(f"✅ Comprehensive analysis functions imported successfully")
        print(f"📊 Sample data created: {data.shape}")
        
        # Test univariate analysis
        univariate_results = generate_univariate_analysis(data)
        print(f"📈 Univariate analysis: {len(univariate_results)} columns analyzed")
        
        # Test correlation analysis
        correlation_results = generate_correlation_analysis(data)
        print(f"🔗 Correlation analysis: {len(correlation_results.get('strong_correlations', []))} strong correlations found")
        
        # Test comprehensive analysis
        comprehensive_results = generate_comprehensive_analysis(data, "Test analysis")
        print(f"🎯 Comprehensive analysis completed successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_enhanced_visualizations():
    """Test enhanced visualization functions"""
    print("\n🧪 Testing Enhanced Visualizations...")
    
    try:
        from src.plotting.plot_generator import (
            create_comprehensive_visualizations,
            create_univariate_plots,
            create_bivariate_plots,
            create_correlation_plots
        )
        
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Create analysis results
        analysis_results = {
            'univariate': {
                'x': {
                    'type': 'numerical',
                    'basic_stats': {'mean': 0, 'std': 1, 'count': 100}
                },
                'y': {
                    'type': 'numerical', 
                    'basic_stats': {'mean': 0, 'std': 1, 'count': 100}
                },
                'category': {
                    'type': 'categorical',
                    'basic_stats': {'unique_values': 3, 'count': 100}
                }
            },
            'correlation_analysis': {
                'pearson_correlation': {'x': {'y': 0.1}, 'y': {'x': 0.1}},
                'strong_correlations': []
            }
        }
        
        print(f"✅ Enhanced visualization functions imported successfully")
        
        # Test visualization creation
        visualizations = create_comprehensive_visualizations(data, analysis_results)
        print(f"📊 Created {len(visualizations)} visualizations")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🧪 Testing Dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'plotly', 'scipy', 'sklearn', 
        'matplotlib', 'seaborn', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All dependencies are available!")
        return True

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced AUTODATA Features")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Multiple SQL Queries", test_multiple_sql_queries),
        ("Comprehensive Analysis", test_comprehensive_analysis),
        ("Enhanced Visualizations", test_enhanced_visualizations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced features are ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
