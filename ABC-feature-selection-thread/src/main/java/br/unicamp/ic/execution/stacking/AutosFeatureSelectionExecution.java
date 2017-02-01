package br.unicamp.ic.execution.stacking;

public class AutosFeatureSelectionExecution extends FeatureSelectionExecution {

	public AutosFeatureSelectionExecution(boolean[] features) {
		super("autos.arff", features, 100, 6, 0.1);
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
