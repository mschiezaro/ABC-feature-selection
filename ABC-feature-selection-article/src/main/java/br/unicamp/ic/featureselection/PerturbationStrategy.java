package br.unicamp.ic.featureselection;

/**
 * Indica algumas possibilidades de execução da seleção de features
 * 
 * @author Mauricio Schiezaro
 */
public enum PerturbationStrategy {
	/**
	 * Indica que na exploração da vizinhança o parâmetro MR que indica quantas
	 * features serão alteradas na composiçãodde um novo conjunto
	 */
	USE_MR,
	/**
	 * Indica que na exploração da vizinhança apenas uma feature será adiciona
	 * ao conjunto da fonte que foi explorada, que será escolhida aleatoriamente
	 */
	CHANGE_ONE_FEATURE
}
