# ML Algorithm Demo for Palantir Hackathon Judges

## Quick Start

Run the demo with:
```bash
python run_demo.py
```

Or run directly:
```bash
python judge_demo.py
```

## What This Demo Shows

This demonstration reveals exactly how our ML algorithm thinks and makes decisions for dining demand optimization. The judges will see:

### 1. **Transparent Decision-Making Process**
- Step-by-step thinking for each prediction
- Feature analysis and weighting
- Clear reasoning behind every decision

### 2. **Real Student Examples**
- Individual student profiles and meal preferences
- Historical vs. estimated data handling
- Class schedule conflicts and free time analysis

### 3. **ML Model Training**
- Feature engineering process
- Model performance metrics (AUC scores)
- Feature importance rankings

### 4. **Demand Prediction**
- Real-time demand curves
- Peak demand identification
- Meal period breakdowns

### 5. **Algorithm Insights**
- What the ML learned from data
- How it makes predictions
- Performance characteristics

## Key Features Demonstrated

- **🧠 ML Thinking**: Shows the algorithm's decision process
- **📊 Data Processing**: Handles 750+ students and 150+ class sections
- **⚡ Real-time**: Sub-second demand computation
- **🔍 Transparency**: Every decision is explainable
- **📈 Visualization**: ASCII charts of demand patterns
- **🎯 Accuracy**: High-performance ML model (AUC > 0.8)

## Demo Structure

1. **Data Loading**: Load students, swipes, and class data
2. **ML Setup**: Initialize services and compute features
3. **Model Training**: Train gradient boosting classifier
4. **Thinking Process**: Show how algorithm makes decisions
5. **Demand Prediction**: Generate demand curves
6. **Insights**: Explain what the ML learned

## Technical Details

- **Model Type**: Gradient Boosting Classifier
- **Features**: 15+ engineered features per prediction
- **Training Data**: Student-bin matrix with weak labels
- **Performance**: < 1 second for full analysis
- **Scalability**: Handles large datasets efficiently

## For Judges

This demo is designed to be:
- **Educational**: Shows ML thinking process
- **Transparent**: Every decision is explainable
- **Impressive**: Demonstrates real ML capabilities
- **Production-ready**: Shows scalable architecture

The algorithm learns from historical dining patterns and class schedules to predict when students are most likely to dine, enabling optimal dining hall staffing and resource allocation.
