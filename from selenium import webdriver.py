from selenium import webdriver
import pandas as pd

# Ganti path dengan lokasi WebDriver Anda
driver_path = '/path/to/your/webdriver'
# Inisialisasi WebDriver (misalnya, menggunakan Chrome)
driver = webdriver.Chrome(executable_path=driver_path)

# URL halaman OLX yang ingin Anda scrape
url = "https://www.olx.co.id/jakarta-dki_g2000007/mobil-bekas_c198?filter=olxautos_listing_eq_true"

# Buka halaman web
driver.get(url)

# Inisialisasi list kosong untuk menyimpan data
arrIds = []
arrModel = []
arrNama = []
arrJenis = []
arrHarga = []
arrTahun = []
arrBensin = []
arrKm = []
arrTransmisi = []

# Dapatkan elemen-elemen yang sesuai dengan kriteria Anda
items = driver.find_elements_by_css_selector("li.EIR5N")

# Mengambil data dari halaman web
for item in items:
    arrIds.append(item.get_attribute("data-id"))
    arrModel.append(item.find_element_by_css_selector("div.g4q3l").text.strip())
    arrNama.append(item.find_element_by_css_selector("span._2tW1I").text.strip())
    arrJenis.append(item.find_element_by_css_selector("span.tjgMj").text.strip())
    arrHarga.append(item.find_element_by_css_selector("span._89yzn").text.strip())
    arrTahun.append(item.find_element_by_css_selector("span.zLvFQ").text.strip())
    arrBensin.append(item.find_element_by_css_selector("span._2xKfz").text.strip())
    arrKm.append(item.find_element_by_css_selector("span._1F5u3").text.strip())
    arrTransmisi.append(item.find_element_by_css_selector("span._2Ij2X").text.strip())

# Menutup WebDriver
driver.quit()

# Membuat DataFrame dari data yang diambil
df = pd.DataFrame({
    'id': arrIds,
    'model': arrModel,
    'nama': arrNama,
    'jenis': arrJenis,
    'harga': arrHarga,
    'tahun': arrTahun,
    'bahan bakar': arrBensin,
    'KM': arrKm,
    'transmisi': arrTransmisi
})

# Menyimpan DataFrame sebagai CSV
df.to_csv('data_olx.csv', index=False)

print("Data telah disimpan sebagai 'data_olx.csv'")
