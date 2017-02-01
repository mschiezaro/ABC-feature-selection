package br.unicamp.ic.execution.knn;

import weka.classifiers.lazy.IBk;

public class AutosFeatureSelectionExecution extends FeatureSelectionExecution {

	public AutosFeatureSelectionExecution(boolean[] features) {
		super("autos.arff", features, 50, 10, 0.01, new IBk());
	}

	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true };
		FeatureSelectionExecution fs = new AutosFeatureSelectionExecution(
				features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
}
