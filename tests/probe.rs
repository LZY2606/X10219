fn eval(src: &str, files: &[(&str, &str)]) -> Result<serde_json::Value, String> {
    let dir = std::env::temp_dir().join(format!(
        "pklr_probe_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, body) in files {
        std::fs::write(dir.join(name), body).unwrap();
    }
    let main = dir.join("main.pkl");
    std::fs::write(&main, src).unwrap();
    let r = pklr::eval_to_json(&main).map_err(|e| e.to_string());
    std::fs::remove_dir_all(&dir).ok();
    r
}

#[test]
fn probe_cycle_backref_missing_field() {
    let r = eval(
        "import \"b.pkl\" as b\nout = b.name\n",
        &[
            ("a.pkl", "import \"b.pkl\" as b\nname = \"A\"\nbname = b.name\n"),
            ("b.pkl", "import \"a.pkl\" as a\nname = \"B\"\n"),
        ],
    );
    println!("RESULT1: {r:?}");
    let r2 = eval(
        "import \"b.pkl\" as b\nout = b.name\n",
        &[
            ("a.pkl", "name = \"A\"\n"),
            ("b.pkl", "import \"a.pkl\" as a\nname = \"B\"\nvia_a = a.name\n"),
        ],
    );
    println!("RESULT2: {r2:?}");
    let r3 = eval(
        "import \"b.pkl\" as b\nout = b.name\n",
        &[
            ("a.pkl", "name = \"A\"\n"),
            ("b.pkl", "import \"a.pkl\" as a\nname = \"B\"\nback = a.ghost\n"),
        ],
    );
    println!("RESULT3: {r3:?}");
}
