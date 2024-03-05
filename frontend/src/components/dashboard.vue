
<template>

<nav class="fixed top-0 z-50 w-full bg-purple-500">
        <div class="px-3 py-3 lg:px-5 lg:pl-3">
          <div class="flex items-center justify-between">
            <div class="flex items-center justify-start rtl:justify-end">
           
               <a href="/" class="flex ms-2 md:me-24">
                <!-- <img src="https://flowbite.com/docs/images/logo.svg" class="h-8 me-3" alt="FlowBite Logo" /> -->
                <span class="self-center text-xl font-semibold sm:text-2xl whitespace-nowrap dark:text-white text-white">Data Analysis</span>
                </a>
            
            </div>
           </div>
        </div>
    </nav> 
    <aside id="logo-sidebar" class="fixed top-0 left-0 z-40 w-64 h-screen pt-20 transition-transform -translate-x-full bg-white border-r border-gray-200 sm:translate-x-0 dark:bg-gray-800 dark:border-gray-700" aria-label="Sidebar">

<div class="h-full px-3 pb-4 overflow-y-auto bg-white dark:bg-gray-800">
    
    <ul class="space-y-2 font-medium">
      <li>
        <a href="/" class="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group">
            <svg class="flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24">
                <path fill-rule="evenodd" d="M11.3 3.3a1 1 0 0 1 1.4 0l6 6 2 2a1 1 0 0 1-1.4 1.4l-.3-.3V19a2 2 0 0 1-2 2h-3a1 1 0 0 1-1-1v-3h-2v3c0 .6-.4 1-1 1H7a2 2 0 0 1-2-2v-6.6l-.3.3a1 1 0 0 1-1.4-1.4l2-2 6-6Z" clip-rule="evenodd"/>
            </svg>
            <span class="flex-1 ms-3 whitespace-nowrap ml-2">Dashboard</span>
        </a>
    </li>
    
    <li>
        <a href="/index" class="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group">
            <svg class="flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 18 18">
                <path d="M6.143 0H1.857A1.857 1.857 0 0 0 0 1.857v4.286C0 7.169.831 8.00 1.857 8.00h4.286A1.857 1.857 0 0 0 8.00 6.143V1.857A1.857 1.857 0 0 0 6.143 0Zm10 0h-4.286A1.857 1.857 0 0 0 10 1.857v4.286C10 7.169 10.831 8.00 11.857 8.00h4.286A1.857 1.857 0 0 0 18 6.143V1.857A1.857 1.857 0 0 0 16.143 0Zm-10 10H1.857A1.857 1.857 0 0 0 0 11.857v4.286C0 17.169.831 18.00 1.857 18.00h4.286A1.857 1.857 0 0 0 8.00 16.143v-4.286A1.857 1.857 0 0 0 6.143 10Zm10 0h-4.286A1.857 1.857 0 0 0 10 11.857v4.286c0 1.026.831 1.857 1.857 1.857h4.286A1.857 1.857 0 0 0 18 16.143v-4.286A1.857 1.857 0 0 0 16.143 10Z"/>
            </svg>
            <span class="flex-1 ms-3 whitespace-nowrap ml-2">Visualization</span>
        </a>
    </li>
    
       <li>
          <a href="/crosstabs" class="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group">
             <svg class="flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 20 20">
                <path d="m17.418 3.623-.018-.008a6.713 6.713 0 0 0-2.4-.569V2h1a1 1 0 1 0 0-2h-2a1 1 0 0 0-1 1v2H9.89A6.977 6.977 0 0 1 12 8v5h-2V8A5 5 0 1 0 0 8v6a1 1 0 0 0 1 1h8v4a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-4h6a1 1 0 0 0 1-1V8a5 5 0 0 0-2.582-4.377ZM6 12H4a1 1 0 0 1 0-2h2a1 1 0 0 1 0 2Z"/>
             </svg>
             <span class="flex-1 ms-3 whitespace-nowrap ml-2">Crosstabs Calculation</span>
             
          </a>
       </li>
       <li>
          <a href="/Ttest" class="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group">
             <svg class="flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 20 18">
                <path d="M14 2a3.963 3.963 0 0 0-1.4.267 6.439 6.439 0 0 1-1.331 6.638A4 4 0 1 0 14 2Zm1 9h-1.264A6.957 6.957 0 0 1 15 15v2a2.97 2.97 0 0 1-.184 1H19a1 1 0 0 0 1-1v-1a5.006 5.006 0 0 0-5-5ZM6.5 9a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9ZM8 10H5a5.006 5.006 0 0 0-5 5v2a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1v-2a5.006 5.006 0 0 0-5-5Z"/>
             </svg>
             <span class="flex-1 ms-3 whitespace-nowrap ml-2">T-test Calculation</span>
          </a>
       </li>
       <li>
          <a href="/correlation" class="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group">
             <svg class="flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 18 20">
                <path d="M17 5.923A1 1 0 0 0 16 5h-3V4a4 4 0 1 0-8 0v1H2a1 1 0 0 0-1 .923L.086 17.846A2 2 0 0 0 2.08 20h13.84a2 2 0 0 0 1.994-2.153L17 5.923ZM7 9a1 1 0 0 1-2 0V7h2v2Zm0-5a2 2 0 1 1 4 0v1H7V4Zm6 5a1 1 0 1 1-2 0V7h2v2Z"/>
             </svg>
             <span class="flex-1 ms-3 whitespace-nowrap ml-2">Correlation</span>
          </a>
       </li>
       <li>
          <a href="/anova" class="flex items-center p-2 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group">
             <svg class="flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 18 16">
                <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M1 8h11m0 0L8 4m4 4-4 4m4-11h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-3"/>
             </svg>
             <span class="flex-1 ms-3 whitespace-nowrap ml-2">Anova</span>
          </a>
       </li>
       
    </ul>
 </div>
</aside>

<div class="p-4 sm:ml-64 bg-purple-50 h-[100vmin] mt-16">





    <div>
    <h1 class="mb-4 text-3xl font-extrabold text-gray-900 dark:text-white md:text-5xl lg:text-6xl"><span class="text-transparent bg-clip-text bg-gradient-to-r to-emerald-600 from-sky-400">Data </span>Analysis App</h1>
    <p class="text-lg font-normal text-gray-500 lg:text-xl dark:text-gray-400 text-center">
    Facilitate qualitative, predictive, and statistical analysis on a large data set.
    </p>
    </div>

<div class="grid grid-cols-5 max-w-full mx-auto">
  <button class="bg-purple-700 h-[100%] ml-2 shadow rounded p-2 hover:bg-green-700">Generate Analysis</button>
  <button class="bg-purple-700 h-[100%] ml-2 shadow rounded p-2 hover:bg-green-700">Export CSV</button>
</div>
<div class="grid grid-cols-12 p-2 max-w-full mx-auto" >
  <!-- Forms will be appended here -->
  
  <div class="bg-white shadow-lg p-5 col-span-11 shadow border flex w-100 gap-2 overflow-auto" id="form_container">
      <!-- Form loop -->
     
      
      <!-- End of form loop -->
  </div>
  <div class="col-span-1">
      <button id="addRowButton" class="bg-purple-700 h-[80%] ml-2 shadow rounded p-2 flex items-center justify-center">Add Row
          <svg class="w-6 h-6 text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none"
              viewBox="0 0 24 24">
              <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M5 12h14m-7 7V5" />
          </svg>
      </button>
    
          <button id="deleteRowButton" class="bg-purple-700 h-[17%] ml-2 shadow rounded p-2 hover:bg-red-700">delete</button>
    
  </div>
  
</div>


 
</div>

</template>

<!-- <script setup>
import layout from '../components/layout.vue';
</script> -->




<style>
body {
  background-color: #F2F4FA;
  color: black;
}

.custom-button {
  background-color: #836FFF;
  /* Add other styles here */
}

.container {
  background-color: #fff;
  padding: 100px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 2500px;
  text-align: center; /* Center the content horizontally */
}

.container img {
  max-width: auto; /* Make the image fit the container size */
  height: auto; /* Maintain aspect ratio */
}

h1, h2 {
  text-align: center;
  margin-bottom: 20px;
}

form {
  margin-bottom: 20px;
}

label {
  display: block;
  margin-bottom: 5px;
}

select, button {
  width: 100%;
  padding: 8px;
  margin-bottom: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  box-sizing: border-box;
}

button {
  background-color: #007bff;
  color: #fff;
  cursor: pointer;
}

button:hover {
  background-color: #0056b3;
}

table {
  border-collapse: collapse;
  width: 100%;
}

th, td {
  border: 1px solid #ddd;
  padding: 8px;
}

th {
  background-color: #f2f2f2;
  font-weight: bold;
}

.result {
  margin-top: 20px;
}
</style>
