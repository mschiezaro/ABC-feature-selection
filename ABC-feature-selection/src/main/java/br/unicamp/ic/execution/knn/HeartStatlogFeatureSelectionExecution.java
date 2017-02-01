package br.unicamp.ic.execution.knn;

import weka.classifiers.lazy.IBk;

public class HeartStatlogFeatureSelectionExecution extends FeatureSelectionExecution {
	
	public HeartStatlogFeatureSelectionExecution(boolean[] features) {
		super("heart-statlog.arff", features, 100, 10, 0.01, new IBk());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true};
		FeatureSelectionExecution fs = new HeartStatlogFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
	
}
