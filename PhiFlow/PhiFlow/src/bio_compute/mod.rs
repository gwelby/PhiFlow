// PhiFlow Bio-Computational Module
// Direct consciousness-DNA interface and biological programming

pub mod dna_interface;
pub mod protein_folding;
pub mod gene_expression;

pub use dna_interface::{DNAInterface, TransductionMethod};
pub use protein_folding::{ProteinFolder, ConformationTarget};
pub use gene_expression::{GeneExpression, ExpressionState};