import joblib
from utils.common_functions import read_yaml
import config
import numpy as np
from flask import Flask , render_template , request



app = Flask(__name__)
# whatever is written below app , it becomes from the Flask application

model = joblib.load(config.MODEL_PATH)

@app.route("/" , methods=["GET","POST"])
# thats is our homepage, for homepage assign /
def index():
    if request.method=="POST":
        lead_time =int(request.form["lead_time"])
        no_of_special_request = int(request.form["no_of_special_request"])
        avg_price_per_room = float(request.form["avg_price_per_room"])
        arrival_month = int(request.form["arrival_month"])
        arrival_date = int(request.form["arrival_date"])
        market_segment_type = int(request.form["market_segment_type"])
        no_of_week_nights = int(request.form["no_of_week_nights"])
        no_of_weekend_nights = int(request.form["no_of_weekend_nights"])
        type_of_meal_plans = int(request.form["type_of_meal_plans"])
        room_type_reserved = int(request.form["room_type_reserved"])
        no_of_previous_cancellations = int(request.form["no_of_previous_cancellations"])
        no_of_previous_bookings_not_canceled = int(request.form["no_of_previous_bookings_not_canceled"])
        no_of_adults = int(request.form["no_of_adults"])
        no_of_children = int(request.form["no_of_children"])
        required_car_parking_space = int(request.form["required_car_parking_space"])
        arrival_year = int(request.form["arrival_year"])
        lead_time = int(request.form["lead_time"])
        repeated_guest = int(request.form["repeated_guest"])


        features = np.array([[no_of_previous_cancellations,no_of_previous_bookings_not_canceled,no_of_adults,no_of_children,no_of_weekend_nights,no_of_week_nights,required_car_parking_space,lead_time,arrival_year,arrival_month,arrival_date,repeated_guest,avg_price_per_room,no_of_special_request,type_of_meal_plans,room_type_reserved,market_segment_type]])

        prediction = model.predict(features)

        return render_template("index.html" , prediction=prediction[0])

    return render_template("index.html" , prediction = None)


if __name__ =="__main__":
    app.run(host="0.0.0.0" , port=5000)

