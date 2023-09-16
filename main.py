from Generator import *
import streamlit as st
import numpy as np
st.title("Login Dulu")
inputs_1 = st.text_input("nama")
inputs_2 = st.text_input("umur")
buttons = st.button("Login")
if buttons:
    st.title(f"SELAMAT ULANGTAHUN {inputs_1}KUU")
    wrt = f'''
    sayanggg kuu selamat ulang tahun ke-{inputs_2} semoga kamu jadi anak yang berbakti
    membanggakan orangtua sama tambah sayang... sama aku oke sayanggg aku mau kasih kamu sebuah
    surat cinta yang khusus buat kamu sayangg .
    '''
    st.write(wrt)
    list_jdl = ['Assalamu alaikum.' , 'sayang ku' , 'malam' , 'sunyi']
    st.write("jika belum muncul di tunggu yaa sayang..")
    st.write('kalo kamu mau surat yang lain kamu bisa klik login nya untuk surat baru yey')
    pilihan = np.random.choice(list_jdl)
    wrt_1 = Generator_text(model, mode= 'sampling', context=pilihan ,num_gen=1000 , temperature=0.1)
    st.write(f"surat cinta untuk : {inputs_1}")
    st.write(wrt_1)
    st.write("by : yudo nidlom firmansyah")
    
