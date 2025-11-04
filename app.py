from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load processed data and encoders
plan_features = pd.read_csv(r"E:\College\Graduation\Recommendation System\Final\processed_plans.csv", index_col='plan_id')
one_hot_encoder = joblib.load(r"E:\College\Graduation\Recommendation System\Final\one_hot_encoderX.pkl")
ordinal_encoder = joblib.load(r"E:\College\Graduation\Recommendation System\Final\ordinal_encoderX.pkl")

# Define the recommendation pipeline
class RecommendationPipeline:
    def __init__(self, plan_features, one_hot_encoder, ordinal_encoder):
        self.plan_features = plan_features
        self.one_hot_encoder = one_hot_encoder
        self.ordinal_encoder = ordinal_encoder

        # Columns used for encoding
        self.nominal_cols = ['anxiety_level']
        self.ordinal_cols = ['Daily_App_Usage', 'Preferred_Content']
    
    def preprocess_user(self, user_input):
        user_df = pd.DataFrame([user_input])
        user_df['Daily_App_Usage'] = user_df['Daily_App_Usage'].map({'من 1 إلى 2 ساعة':'1-2 hours','من 3 إلى 4 ساعة':'3-4 hours','من 4 إلى 6 ساعة':'4-6 hours','أكثر من 6 ساعات':'More than 6 hours'})
        user_df['Preferred_Content'] = user_df['Preferred_Content'].map({'مقاطع فيديو':'Motivational videos','تمارين تفاعلية':'Interactive exercises','جلسات إرشادية (بودكاست)':'Guided sessions (Podcasts)','قراءة (مقالات)':'Reading (Articles)'})
        user_df['anxiety_level'] = user_df['anxiety_level'].map({'منخفض':'Low','متوسط':'Moderate','عالي':'High'})


        # Check that all required columns are present
        missing_cols = [col for col in self.nominal_cols + self.ordinal_cols if col not in user_df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected input column(s): {', '.join(missing_cols)}")

        # Encode nominal columns using one-hot encoder
        nominal_encoded = self.one_hot_encoder.transform(user_df[self.nominal_cols])
        nominal_feature_names = self.one_hot_encoder.get_feature_names_out(self.nominal_cols)
        encoded_nominal_df = pd.DataFrame(nominal_encoded, columns=nominal_feature_names)

        # Encode ordinal columns
        ordinal_encoded = self.ordinal_encoder.transform(user_df[self.ordinal_cols])
        encoded_ordinal_df = pd.DataFrame(ordinal_encoded, columns=self.ordinal_cols)

        # Combine encoded features
        encoded_user = pd.concat([encoded_nominal_df, encoded_ordinal_df], axis=1)

        # Align columns with plan_features and fill missing columns with zeros
        encoded_user = encoded_user.reindex(columns=self.plan_features.columns, fill_value=0)

        return encoded_user

    def recommend(self, user_input):
        encoded_user = self.preprocess_user(user_input)
        similarity_scores = cosine_similarity(encoded_user, self.plan_features)[0]
        top_index = similarity_scores.argmax()
        return self.plan_features.index[top_index]

# Initialize pipeline
pipeline = RecommendationPipeline(plan_features, one_hot_encoder, ordinal_encoder)

# Set up Flask app
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        raw_user_data = request.get_json()
        print("Received input:", raw_user_data)  # Debug: Log input

        recommended_plan_id = pipeline.recommend(raw_user_data)
        return jsonify({'recommended_plan_id': int(recommended_plan_id)})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred: ' + str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
