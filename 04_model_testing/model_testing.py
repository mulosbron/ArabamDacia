import pandas as pd
import joblib
import os


def get_valid_input(prompt, type_func, condition, error_msg):
    while True:
        try:
            value = type_func(input(prompt))
            if not condition(value):
                print(error_msg)
                continue
            return value
        except Exception as e:
            print("Invalid input. Please enter a value of the correct type.")
            print(e)


def main():
    print("Welcome to Price Prediction Application!")

    age = get_valid_input(
        "Vehicle Age: ",
        int,
        lambda x: x >= 0,
        "Vehicle age cannot be negative."
    )

    mileage = get_valid_input(
        "Mileage: ",
        int,
        lambda x: x >= 0,
        "Mileage cannot be negative."
    )

    avg_fuel_consumption = get_valid_input(
        "Average Fuel Consumption (e.g.: 4.0): ",
        float,
        lambda x: 3.0 <= x <= 10.0,
        "Fuel consumption must be between 3 and 10."
    )

    transmission_type = get_valid_input(
        "Transmission Type (0 = Manual, 1 = Automatic, 2 = Semi-Automatic): ",
        int,
        lambda x: x in [0, 1, 2],
        "Transmission Type can only be 0, 1, or 2."
    )

    changed = get_valid_input(
        "Number of Changed Parts (e.g.: 0, 1, 2): ",
        int,
        lambda x: x >= 0,
        "Number of changed parts cannot be negative."
    )

    painted = get_valid_input(
        "Number of Painted Parts (e.g.: 0, 1, 2): ",
        int,
        lambda x: x >= 0,
        "Number of painted parts cannot be negative."
    )

    model_path = '../03_model_training/best_model.pkl'

    if not os.path.exists(model_path):
        print("Model file not found. Please check the file.")
        return

    model = joblib.load(model_path)

    user_input = {
        'mileage': mileage,
        'avg_fuel_consumption': avg_fuel_consumption,
        'age': age,
        'mileage/age': (mileage / age),
        'transmission_type_Düz': 1 if transmission_type == 0 else 0,
        'transmission_type_Otomatik': 1 if transmission_type == 1 else 0,
        'transmission_type_Yarı Otomatik': 1 if transmission_type == 2 else 0,
        'changed': changed,
        'painted': painted
    }

    df = pd.DataFrame([user_input])

    df = df[['mileage', 'avg_fuel_consumption', 'age', 'mileage/age',
             'transmission_type_Düz', 'transmission_type_Otomatik', 'transmission_type_Yarı Otomatik',
             'changed', 'painted']]

    try:
        price_pred = model.predict(df)[0]
        print(f"\nPredicted Price: {price_pred:,.2f} Turkish Lira")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()
