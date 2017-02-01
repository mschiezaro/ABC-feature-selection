package br.unicamp.ic.execution.knn;

import weka.classifiers.lazy.IBk;

public class DiabetesFeatureSelectionExecution extends FeatureSelectionExecution {

	public DiabetesFeatureSelectionExecution(boolean[] features) {
		super("diabetes.arff", features, 100, 10 , 0.2, new IBk());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true};		
		FeatureSelectionExecution fs = new DiabetesFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
	
}
