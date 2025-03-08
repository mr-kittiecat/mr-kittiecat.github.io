let model;

// Modell bei Seitenladen laden
fetch('ridge_model.json').then(res => res.json()).then(loadedModel => {
    model = loadedModel;
    console.log("✅ Modell geladen!", model);
});

// Formular-Eventlistener
document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();

    // Eingaben abrufen
    const brand = parseInt(document.getElementById('brand').value);
    const engine_size = parseFloat(document.getElementById('engine_size').value);
    const car_age = parseFloat(document.getElementById('car_age').value);
    const mileage = parseFloat(document.getElementById('mileage').value);
    const fuel_type = parseInt(document.getElementById('fuel_type').value);
    const transmission = parseInt(document.getElementById('transmission').value);
    const doors = parseInt(document.getElementById('doors').value);

    // Features vorbereiten
    let features = [brand, engine_size, Math.sqrt(mileage), car_age, fuel_type, transmission, doors];

    // Standardisierung durchführen
    let scaled_features = features.map((x, i) => 
        (x - model.scaler_mean[i]) / model.scaler_scale[i]
    );

    // Polynomial Features generieren
    const polyFeatures = [...scaled_features];
    polyFeatures.push(...scaled_features.map(x => x**2));
    polyFeatures.push(...scaled_features.map(x => x**3));

    // Vorhersage mit Ridge-Koeffizienten berechnen
    let prediction = model.ridge_intercept;
    model.ridge_coefficients.forEach((coef, idx) => {
        prediction += coef * polyFeatures[idx];
    });

    // Preis rücktransformieren aus log-Skala
    let predictedPrice = Math.expm1(prediction);

    // Ergebnis anzeigen
    document.getElementById('predicted-price').innerText = predictedPrice.toFixed(2);
});
