package br.unicamp.ic.execution.svm;

import weka.classifiers.functions.LibSVM;

public class ZooFeatureSelectionExecution extends FeatureSelectionExecution {

	public ZooFeatureSelectionExecution(boolean[] features) {
		super( "zoo.arff", features, 100, 6, 0.1, new LibSVM());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true};
		FeatureSelectionExecution fs = new ZooFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
}
