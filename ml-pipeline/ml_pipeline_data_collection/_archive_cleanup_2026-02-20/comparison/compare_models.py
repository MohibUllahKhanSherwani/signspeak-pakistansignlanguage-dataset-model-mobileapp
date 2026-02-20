"""
Model Comparison Script

Trains both baseline and augmented models, then compares them.
Saves detailed comparison report.
"""

import numpy as np
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import json

from actions_config import load_actions, DATA_PATH, SEQUENCE_LENGTH, AUGMENTATION_MULTIPLIER, EPOCHS
from train_model import load_data as load_data_simple, build_model as build_model_simple
from train_model_with_augmentation import load_data, build_model
from data_augmentation import create_augmented_dataset
from training_logger import log_training_session, log_comparison_session
import time

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_and_evaluate(actions, use_augmentation=False, augment_multiplier=3):
    """
    Train a model and return evaluation metrics.
    
    Args:
        actions: List of action names
        use_augmentation: Whether to use augmentation
        augment_multiplier: Augmentation multiplier
    
    Returns:
        dict with metrics and model info
    """
    print("\n" + "=" * 70)
    mode = "WITH AUGMENTATION" if use_augmentation else "BASELINE (No Augmentation)"
    print(f"Training Model: {mode}")
    print("=" * 70)
    
    # Load data
    X, y = load_data(actions)
    print(f"Loaded {len(X)} sequences")
    
    # Apply augmentation if requested
    if use_augmentation:
        print(f"Applying {augment_multiplier}x augmentation...")
        X, y = create_augmented_dataset(X, y, augmentation_multiplier=augment_multiplier)
        print(f"Augmented dataset: {len(X)} sequences")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded).astype(int)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, y_cat.shape[1], use_dropout=use_augmentation)
    
    # Train
    print("\nTraining...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,  # Use config epochs
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )
    training_duration = time.time() - start_time
    
    # Evaluate
    print("\nEvaluating...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Classification report
    report = classification_report(
        y_test_classes, y_pred_classes,
        target_names=actions,
        output_dict=True
    )
    
    # Save model
    model_name = "action_model_augmented.h5" if use_augmentation else "action_model_baseline.h5"
    encoder_name = "label_encoder_augmented.pkl" if use_augmentation else "label_encoder_baseline.pkl"
    
    model.save(model_name)
    joblib.dump(le, encoder_name)
    
    print(f"\n✅ Saved: {model_name}, {encoder_name}")
    
    # Log the session
    log_training_session(
        duration_seconds=training_duration,
        num_words=len(actions),
        training_acc=train_acc,
        val_acc=test_acc,
        epochs=len(history.history['accuracy']),
        batch_size=16,
        augmented=use_augmentation,
        model_path=model_name
    )
    
    return {
        'mode': mode,
        'use_augmentation': use_augmentation,
        'augment_multiplier': augment_multiplier if use_augmentation else 1,
        'num_sequences': len(X),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_loss': float(train_loss),
        'test_loss': float(test_loss),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'model_file': model_name,
        'encoder_file': encoder_name,
        'history': {
            'train_acc': [float(x) for x in history.history['accuracy']],
            'val_acc': [float(x) for x in history.history['val_accuracy']],
            'train_loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
        },
        'training_duration': training_duration
    }


def compare_models(baseline_results, augmented_results):
    """
    Generate comparison report.
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT")
    print("=" * 70)
    
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dataset sizes
    print("\n📊 DATASET SIZES:")
    print(f"{'Metric':<30} {'Baseline':<20} {'Augmented':<20}")
    print("-" * 70)
    print(f"{'Training sequences':<30} {baseline_results['num_sequences']:<20} {augmented_results['num_sequences']:<20}")
    print(f"{'Multiplier':<30} {baseline_results['augment_multiplier']:<20} {augmented_results['augment_multiplier']:<20}")
    
    # Accuracy comparison
    print("\n🎯 ACCURACY COMPARISON:")
    print(f"{'Metric':<30} {'Baseline':<20} {'Augmented':<20} {'Difference':<20}")
    print("-" * 90)
    
    baseline_train = baseline_results['train_accuracy'] * 100
    augmented_train = augmented_results['train_accuracy'] * 100
    train_diff = augmented_train - baseline_train
    
    baseline_test = baseline_results['test_accuracy'] * 100
    augmented_test = augmented_results['test_accuracy'] * 100
    test_diff = augmented_test - baseline_test
    
    print(f"{'Training Accuracy':<30} {baseline_train:<20.2f}% {augmented_train:<20.2f}% {train_diff:+.2f}%")
    print(f"{'Test Accuracy':<30} {baseline_test:<20.2f}% {augmented_test:<20.2f}% {test_diff:+.2f}%")
    
    # Overfitting check
    baseline_gap = baseline_train - baseline_test
    augmented_gap = augmented_train - augmented_test
    gap_diff = baseline_gap - augmented_gap
    
    print(f"{'Train-Test Gap (overfitting)':<30} {baseline_gap:<20.2f}% {augmented_gap:<20.2f}% {gap_diff:+.2f}%")
    
    # Analysis
    print("\n💡 ANALYSIS:")
    
    if test_diff > 5:
        print(f"✅ Augmented model has {test_diff:.1f}% higher test accuracy - SIGNIFICANT IMPROVEMENT!")
    elif test_diff > 0:
        print(f"✅ Augmented model has {test_diff:.1f}% higher test accuracy - Slight improvement")
    else:
        print(f"⚠️  Baseline model has higher test accuracy - Consider collecting more data")
    
    if augmented_gap < baseline_gap:
        improvement = baseline_gap - augmented_gap
        print(f"✅ Augmentation reduced overfitting by {improvement:.1f}% - Better generalization!")
    
    if baseline_gap > 15:
        print(f"⚠️  Baseline model shows significant overfitting ({baseline_gap:.1f}% gap)")
    
    if augmented_gap > 10:
        print(f"⚠️  Augmented model still shows some overfitting ({augmented_gap:.1f}% gap)")
        print(f"   💡 Consider: More augmentation or more data collection")
    
    # Recommendation
    print("\n🏆 RECOMMENDATION:")
    if test_diff > 2 or augmented_gap < baseline_gap - 3:
        print("   ✅ USE AUGMENTED MODEL for deployment")
        print(f"   📁 File: {augmented_results['model_file']}")
    elif test_diff < -5:
        print("   ⚠️  USE BASELINE MODEL (augmentation didn't help)")
        print(f"   📁 File: {baseline_results['model_file']}")
    else:
        print("   Both models perform similarly - either is fine")
        print(f"   📁 Augmented: {augmented_results['model_file']}")
    
    # Per-class accuracy
    print("\n📋 PER-CLASS ACCURACY:")
    print(f"{'Sign':<20} {'Baseline':<15} {'Augmented':<15} {'Difference':<15}")
    print("-" * 70)
    
    for action in baseline_results['classification_report'].keys():
        if action in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        
        baseline_f1 = baseline_results['classification_report'][action].get('f1-score', 0) * 100
        augmented_f1 = augmented_results['classification_report'][action].get('f1-score', 0) * 100
        diff = augmented_f1 - baseline_f1
        
        print(f"{action:<20} {baseline_f1:<15.2f}% {augmented_f1:<15.2f}% {diff:+.2f}%")
    
    return {
        'baseline': baseline_results,
        'augmented': augmented_results,
        'comparison': {
            'test_accuracy_diff': test_diff,
            'overfitting_reduction': baseline_gap - augmented_gap,
            'recommended_model': 'augmented' if test_diff > 2 else 'baseline'
        }
    }


def main():
    total_start_time = time.time()
    print("=" * 70)
    print("MODEL COMPARISON: Baseline vs Augmented")
    print("=" * 70)
    
    # Load actions
    actions = load_actions()
    print(f"\nActions: {', '.join(actions)}")
    
    print("\nStarting training (this will take time)...")
    
    # Train baseline model
    baseline_results = train_and_evaluate(actions, use_augmentation=False)
    
    print("\n" + "~" * 70)
    print("Baseline done! Moving to augmented model training...")
    
    # Train augmented model
    augmented_results = train_and_evaluate(actions, use_augmentation=True, augment_multiplier=AUGMENTATION_MULTIPLIER)
    
    # Compare
    comparison = compare_models(baseline_results, augmented_results)
    
    # Save report
    report_dir = "comparison_reports"
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    total_duration = time.time() - total_start_time
    comparison['total_execution_time_seconds'] = total_duration
    
    with open(report_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_file}")
    
    # Record total comparison time
    log_comparison_session(
        total_duration=total_duration,
        baseline_duration=baseline_results['training_duration'],
        augmented_duration=augmented_results['training_duration'],
        num_words=len(actions),
        baseline_acc=baseline_results['test_accuracy'],
        augmented_acc=augmented_results['test_accuracy']
    )
    
    mins = int(total_duration // 60)
    secs = int(total_duration % 60)
    print(f"⏱️  Total Session Time: {mins}m {secs}s")
    
    # Print breakdown
    b_min, b_sec = int(baseline_results['training_duration'] // 60), int(baseline_results['training_duration'] % 60)
    a_min, a_sec = int(augmented_results['training_duration'] // 60), int(augmented_results['training_duration'] % 60)
    print(f"   • Baseline Training: {b_min}m {b_sec}s")
    print(f"   • Augmented Training: {a_min}m {a_sec}s")
    
    print("\n Comparison complete!")

if __name__ == "__main__":
    main()
