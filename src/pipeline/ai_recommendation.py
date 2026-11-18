class AIRecommendation:

    @staticmethod
    def generate(
        prediction,
        age,
        balance,
        tenure,
        is_active_member,
        num_products
    ):

        recommendations = []

        if prediction == 1:

            recommendations.append(
                "Customer is at risk of churn."
            )

            if balance > 100000:

                recommendations.append(
                    "Assign a relationship manager."
                )

            if is_active_member == 0:

                recommendations.append(
                    "Launch a re-engagement campaign."
                )

            if tenure < 3:

                recommendations.append(
                    "Provide onboarding incentives."
                )

            if num_products <= 1:

                recommendations.append(
                    "Recommend additional banking products."
                )

            if age > 50:

                recommendations.append(
                    "Offer tailored loyalty benefits."
                )

        else:

            recommendations.append(
                "Customer appears loyal."
            )

            recommendations.append(
                "Maintain satisfaction through regular engagement."
            )

        return recommendations