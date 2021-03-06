package br.unicamp.ic.execution.stacking;

public class HepaticFeatureSelectionExecution extends FeatureSelectionExecution {

	public HepaticFeatureSelectionExecution(boolean[] features) {
		super("hepatitis.arff", features, 100, 6, 0.1);
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true };		
		FeatureSelectionExecution fs = new HepaticFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
	}
}
