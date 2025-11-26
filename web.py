import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, uniform, gamma, chi2, f, laplace, logistic
from scipy.stats import weibull_min, t, pareto, lognorm, rayleigh, cauchy, beta
import numpy as np


#startup propmts
st.title("Histogram Creator")
st.write("Would you like to upload a file or hand type your data?")
choice = st.text_input("Enter either 'file' or 'typed'")

#global variable decleration
xaxis = str
yaxis = str
title = str
clock = 0
values = pd.DataFrame()

#gets the data from the user
if choice.upper() == "FILE":
    title = '"Your Graph'
    title = st.text_input("Enter your graphs title:")
    inter = st.file_uploader("Upload your file: ")    
    if inter != None:
        data = pd.read_csv(inter)
        st.dataframe(data) 
        taxis = data.columns.tolist()
        xaxis = taxis[0]
        yaxis = taxis[1]
        values = data
        clock = 1
        done = True
elif choice.upper() == "TYPED":        
    xaxis = st.text_input("Enter you x-axis name:")
    yaxis = st.text_input("Enter the y-axis name:")
    title = st.text_input("Add a title to your data:")
    df = pd.DataFrame({xaxis: [None], yaxis: [None],})
    values = st.data_editor(df, num_rows="dynamic")
    done = st.button("Done")
    clock = 1
elif choice != "" or choice == "None":
    st.write('Not a valid entry, try again')
       
#  draws the initial graphs
if clock == 1 and done:
    fig, sct = plt.subplots()
    sct.scatter(values[xaxis], values[yaxis])
    sct.set_xlabel(xaxis)
    sct.set_ylabel(yaxis)
    sct.set_title("Your Data")
    st.pyplot(fig)
        
    fig, hst = plt.subplots()
    hst.hist(values, bins=25, density=True)
    hst.set_title(title)
    hst.set_xlabel(xaxis)
    hst.set_ylabel(yaxis)
    st.pyplot(fig)
    clock = 2

#formats the page
buttons = st.container()
graph = st.container()
 
#decides the distribution
if clock == 2:    
    with buttons:
        st.write("Would like to add a type of distribution? (Please only select one)")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.toggle("Normal"):
                dist = norm
                clock = 3
                dtype = 1
            if st.toggle("Exponential"):
                clock = 3
                dist = expon
                dtype = 1
            if st.toggle("Rayleigh"):
                clock = 3
                dist = rayleigh
                dtype = 1
        with col2:
            if st.toggle("Gamma"):
                clock = 3
                dist = gamma
                dtype = 2
            if st.toggle("Weibull"):
                clock = 3
                dtype = 2
                dist = weibull_min
            if st.toggle("Cauchy"):
                clock = 3
                dist = cauchy
                dtype = 1
        with col3:
            if st.toggle("Beta"):
                clock = 3
                dtype = 3
                dist = beta
            if st.toggle("Uniform"):
                clock = 3
                dtype = 1
                dist = uniform
            if st.toggle("Laplace"):
                clock = 3
                dtype = 1
                dist = laplace
        with col4:
            if st.toggle("Log Normal"):
                clock = 3
                dist = lognorm
                dtype = 2
            if st.toggle("Chi-Square"):
                clock = 3
                dist = chi2
                dtype = 2
            if st.toggle("Logistic"):
                clock = 3
                dtype = 1
                dist = logistic
        with col5:
            if st.toggle("Student-T"):
                clock = 3
                dtype = 2
                dist = t
            if st.toggle("Pareto"):
                clock = 3
                dtype = 4
                dist = pareto
            if st.toggle("F-Dist"):
                clock = 3
                dtype = 3
                dist = f
                
#lets the user mess with the graph     
if clock == 3:
    with graph:
        st.markdown("#### Parameters:")
        if dtype == 1:
            data = values[yaxis].dropna()
            mu, sigma = dist.fit(data)
            x = np.linspace(data.min(), data.max(), 300)
            usermu = st.slider("Mean", float(data.max())*-1.5, float(data.max())*3, mu)
            usersigma = st.slider("Standard Deviation", float(data.max())*-1.5, float(data.max())*3, sigma)
        
            pdf = dist.pdf(x, usermu, usersigma)    
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, density=True)
            ax.plot(x, pdf, linewidth=2)

            ax.set_xlabel(xaxis)
            ax.set_ylabel(yaxis)
            ax.set_title("Adjustable Distribution")
            ax.legend()

            st.pyplot(fig)
        elif dtype == 2:
            data = values[yaxis].dropna()
            a, loc, scale = dist.fit(data)
            x = np.linspace(data.min(), data.max(), 300)
                        
            if a < 250:
                usera = st.slider("Shape", float(data.max())*-1.5, float(data.max())*3, float(round(a)))
            else:
                usera = st.slider("Shape", float(data.max())*-1.5, float(data.max())*3, 200.0)
            userloc = st.slider("Location", float(data.max())*-1.5, float(data.max())*3, float(round(loc)))
            userscale = st.slider("Scale", float(data.max())*-1.5, float(data.max())*3, float(scale))
        
            pdf = dist.pdf(x, usera, userloc, userscale)
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, density=True)
            ax.plot(x, pdf, linewidth=2)

            ax.set_xlabel(xaxis)
            ax.set_ylabel(yaxis)
            ax.set_title("Adjustable Distribution")
            ax.legend()

            st.pyplot(fig)
        elif dtype == 3:
            data = values[yaxis].dropna()
            a, b, loc, scale = beta.fit(data)            
            x = np.linspace(data.min(), data.max(), 300)
            
            userb = st.slider("Left Skew", float(data.max())*-1.5, float(data.max())*3, float(b))
            usera = st.slider("Right Skew", float(data.max())*-1.5, float(data.max())*3, float(round(a)))
            userloc = st.slider("Location", float(data.max())*-1.5, float(data.max())*3, float(round(loc)))
            userscale = st.slider("Scale", float(data.max())*-1.5, float(data.max())*3, float(scale))
            
            pdf = dist.pdf(x, usera, userb, userloc, userscale)
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, density=True)
            ax.plot(x, pdf, linewidth=2)

            ax.set_xlabel(xaxis)
            ax.set_ylabel(yaxis)
            ax.set_title("Adjustable Distribution")
            ax.legend()

            st.pyplot(fig)
        elif dtype == 4:
            data = values[yaxis].dropna()
            a, loc, scale = pareto.fit(data)
            xmin = data.min()
            if loc >= xmin:
                loc = xmin - 0.001
            x = np.linspace(xmin, data.max(), 300)

            usera = st.slider("Shape", 0.1, 10.0, float(min(max(a, 0.1), 10.0)))
            userloc = st.slider("Location", float(xmin - abs(xmin) * 2), xmin - 0.001, float(loc))
            userscale = st.slider("Scale", 0.001,  float(abs(data.max()) * 3), float(scale + 0.1))

            pdf = pareto.pdf(x, usera, userloc, userscale)
            fig, ax = plt.subplots()
            ax.hist(data, bins=30, density=True)
            ax.plot(x, pdf, linewidth=2)

            ax.set_xlabel(xaxis)
            ax.set_ylabel(yaxis)
            ax.set_title("Adjustable Distribution")
            ax.legend()
            st.pyplot(fig)
        
#calculates and displays the margin or error
        if dtype == 1:
            #MSE
            histval, bine = np.histogram(data, bins=30, density=True)
            binmid = (bine[:-1] + bine[1:]) / 2
            pdfval = dist.pdf(binmid, usermu, usersigma)
            mse = np.mean((histval - pdfval) ** 2)
            
            #Max error
            err = np.max(np.abs(histval - pdfval))
        elif dtype == 2 or dtype == 4:
            #MSE
            histval, bine = np.histogram(data, bins=30, density=True)
            binmid = (bine[:-1] + bine[1:]) / 2
            pdfval = dist.pdf(binmid, usera, userloc, userscale)
            mse = np.mean((histval - pdfval) ** 2)
            
            #Max error
            err = np.max(np.abs(histval - pdfval))
        elif dtype == 3:
            #MSE
            histval, bine = np.histogram(data, bins=30, density=True)
            binmid = (bine[:-1] + bine[1:]) / 2
            pdfval = dist.pdf(binmid, usera, userb, userloc, userscale)
            mse = np.mean((histval - pdfval) ** 2)
            
            #Max error
            err = np.max(np.abs(histval - pdfval))
        
        ncol1, ncol2 = st.columns(2)
        with ncol1:
            st.metric("Maximum Error", round(err, 6))  
        with ncol2:
            st.metric("Mean Squared Error (hist vs PDF)", round(mse, 6))
        
        clock = 4

if clock == 4:
    st.header("Thank you for using!")