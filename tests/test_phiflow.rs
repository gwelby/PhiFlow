#[cfg(test)]
mod tests {
    use quantum_core::quantum::run_phiflow_demo;

    #[test]
    fn test_phiflow_demo() {
        let result = run_phiflow_demo();
        println!("{}", result);
        assert!(!result.is_empty());
    }
}
