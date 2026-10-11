import streamlit as st

st.set_page_config(page_title="Customer Churn Demo")
st.title("Customer Churn Demo")
st.write("Enter a few customer details to preview the profile.")

with st.form("customer_form"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input(
        "Monthly charges ($)", min_value=0.0, value=50.0, step=5.0
    )
    submitted = st.form_submit_button("Show profile")

if submitted:
    st.subheader("Customer profile")
    st.write(f"Tenure: {tenure} months")
    st.write(f"Monthly charges: ${monthly_charges:.2f}")
    st.write(f"Estimated total charges: ${tenure * monthly_charges:.2f}")