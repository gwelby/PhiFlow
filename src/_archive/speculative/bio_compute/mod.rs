// PhiFlow Bio-Computational Module
// Direct consciousness-DNA interface and biological programming

pub mod dna_interface;
pub mod gene_expression;
pub mod protein_folding;

pub use dna_interface::{DNAInterface, TransductionMethod};
pub use gene_expression::{ExpressionState, GeneExpression};
pub use protein_folding::{ConformationTarget, ProteinFolder};
