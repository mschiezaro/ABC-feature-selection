package br.unicamp.ic.featureselection.swarm;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Representa uma fonte de alimento
 * 
 * @author Mauricio Schiezaro
 * 
 */
public class FoodSource implements Comparable<FoodSource>, Serializable {

	private static final long serialVersionUID = -6165043365329906757L;

	/**
	 * Vetor que representa as features que serão utilizadas na classificação.
	 * Cada posição do vetor indica uma feature e o valor true indica que ela
	 * participará da classificação, false não participará
	 */
	private boolean featureInclusion[];

	/**
	 * Qualidade da fonte de alimento, representada pela acurácia do
	 * classificador utilizando as features indicaas por featureInclusion
	 */
	private double fitness;
	
	/**
	 * Número de vezes que essa fonte foi visitada e não foi achada nenhuma
	 * fonte mais rica na vizinhança
	 */
	private int limit;

	/**
	 * Número de features utilizadas por essa fonte de alimento
	 */
	private int nrFeatures = 0;

	public FoodSource(boolean featureAInclusion[]) {
		this.featureInclusion = featureAInclusion;
	}
	
	public FoodSource() {
		
	}

	/**
	 * 
	 * @param featureAInclusion
	 *            * Vetor que representa as features que serão utilizadas na
	 *            classificação. Cada posição do vetor indica uma feature e o
	 *            valor true indica que ela participará da classificação, false
	 *            não participará
	 * @param fitness
	 *            Qualidade da fonte de alimento, representada pela acurácia do
	 *            classificador utilizando as features indicaas por
	 *            featureInclusion
	 * @param nrFeatures
	 *            Número de features utilizadas por essa fonte de alimento
	 */
	public FoodSource(boolean featureAInclusion[], double fitness,
			int nrFeatures) {
		this.featureInclusion = featureAInclusion;
		this.fitness = fitness;
		limit = 0;
		this.nrFeatures = nrFeatures;
	}

	/**
	 * 
	 * @param foodSource
	 *            Fonte de alimento que será feita uma cópia
	 */
	public FoodSource(FoodSource foodSource) {
		fitness = foodSource.fitness;
		limit = foodSource.limit;
		boolean features[] = foodSource.featureInclusion;
		featureInclusion = Arrays.copyOf(features, features.length);
		nrFeatures = foodSource.nrFeatures;
	}

	/**
	 * Retorna uma cópia do Vetor que representa as features que serão
	 * utilizadas na classificação. Cada posição do vetor indica uma feature e o
	 * valor true indica que ela participará da classificação, false não
	 * participará
	 * 
	 * @return Cópia do Vetor que representa as features que serão utilizadas na
	 *         classificação. Cada posição do vetor indica uma feature e o valor
	 *         true indica que ela participará da classificação, false não
	 *         participará
	 */
	public boolean[] getFeatureInclusion() {
		return Arrays.copyOf(featureInclusion, featureInclusion.length);
	}

	/**
	 * Atualiza o Vetor que representa as features que serão utilizadas na
	 * classificação. Cada posição do vetor indica uma feature e o valor true
	 * indica que ela participará da classificação, false não participará
	 * 
	 * @param featureInclusion
	 *            Vetor que representa as features que serão utilizadas na
	 *            classificação. Cada posição do vetor indica uma feature e o
	 *            valor true indica que ela participará da classificação, false
	 *            não participará
	 */
	public void setFeatureInclusion(boolean[] featureInclusion) {
		this.featureInclusion = featureInclusion;
	}

	/**
	 * Retorna a qualidade da fonte de alimento, representada pela acurácia do
	 * classificador utilizando as features indicaas por featureInclusion
	 * 
	 * @return A qualidade da fonte de alimento, representada pela acurácia do
	 *         classificador utilizando as features indicaas por
	 *         featureInclusion
	 */
	public double getFitness() {
		return fitness;
	}

	/**
	 * Atualiza a qualidade da fonte de alimento, representada pela acurácia do
	 * classificador utilizando as features indicaas por featureInclusion
	 * 
	 * @param fitness
	 *            A qualidade da fonte de alimento, representada pela acurácia
	 *            do classificador utilizando as features indicaas por
	 *            featureInclusion
	 */
	public void setFitness(double fitness) {
		this.fitness = fitness;
	}
	
	/**
	 * Retorna o número de vezes que essa fonte foi visitada e não foi achada
	 * nenhuma fonte mais rica na vizinhança
	 * 
	 * @return Número de vezes que essa fonte foi visitada e não foi achada
	 *         nenhuma fonte mais rica na vizinhança
	 */
	public int getLimit() {
		return limit;
	}

	/**
	 * Atualiza o número de vezes que essa fonte foi visitada e não foi achada
	 * nenhuma fonte mais rica na vizinhança
	 * 
	 * @param limit
	 *            Número de vezes que essa fonte foi visitada e não foi achada
	 *            nenhuma fonte mais rica na vizinhança
	 */
	public void setLimit(int limit) {
		this.limit = limit;
	}

	/**
	 * Incrementa o número de vezes que essa fonte foi visitada e não foi achada
	 * nenhuma fonte mais rica na vizinhança
	 */
	public void incrementLimit() {
		limit++;
	}

	/**
	 * Retorna o número de features utilizadas por essa fonte de alimento
	 * 
	 * @return Número de features utilizadas por essa fonte de alimento
	 */
	public int getNrFeatures() {
		return nrFeatures;
	}

	/**
	 * Atualiza o número de features utilizadas por essa fonte de alimento
	 * 
	 * @param nrFeatures
	 *            Número de features utilizadas por essa fonte de alimento
	 */
	public void setNrFeatures(int nrFeatures) {
		this.nrFeatures = nrFeatures;
	}

	/**
	 * Incrementa o número de features utilizadas por essa fonte de alimento
	 */
	public void increaseNrFeatures() {
		nrFeatures++;
	}

	/**
	 * Para identificar uma igualdade só verificamos se uma fonte tem o mesmo
	 * vetor de features
	 * 
	 * @return true se as fontes de alimento possuem o mesmo vetor de featuresm
	 *         false caso o contrário
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		FoodSource other = (FoodSource) obj;
		if (!Arrays.equals(featureInclusion, other.featureInclusion))
			return false;
		return true;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(featureInclusion);
		return result;
	}

	@Override
	public String toString() {
		return "FoodSource [ features = " + Arrays.toString(featureInclusion)
				+ ", fitness = " + fitness + " %, limit = " + limit
				+ ", nrFeatures = " + nrFeatures + " ]";
	}

	@Override
	public int compareTo(FoodSource foodSource) {
		double res = fitness - foodSource.fitness;
		if (res < 0) {
			return -1;
		} else if (res > 0) {
			return 1;
		}
		return 0;
	}
}
