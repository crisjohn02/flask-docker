<template>
  
  <div>
    <!-- Sidebar -->
    <aside :class="sidebarClasses">
      <!-- Sidebar Content -->
      <a href="/" class="block text-center py-10">
        <span class="self-center text-xl font-semibold sm:text-2xl whitespace-nowrap dark:text-white text-black mb-6">Data Analysis</span>
      </a>
      <div class="h-full px-3 pb-4 overflow-y-auto">
        <ul class="space-y-2 font-medium text-left">
          <li v-for="link in links" :key="link.id">
            <a :href="link.url" class="flex p-2 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group">
              <svg :class="link.iconClasses" aria-hidden="true" fill="currentColor" viewBox="0 0 24 24">
                <path :d="link.iconPath"/>
              </svg>
              <span class="flex-1 ms-3 whitespace-nowrap ml-2">{{ link.label }}</span>
            </a>
          </li>
        </ul>
      </div>
    </aside>
    
    <!-- Main Content -->
    <div :class="mainContentClasses">
      
      <nav class="fixed top-0 z-50 w-full bg-purple-500">
        
        <div class="flex items-center justify-start rtl:justify-end">
          
          <div>
            <button @click="toggleMenu" class="hamburger">
              <span class="hamburger-box">
                <span class="hamburger-inner"></span>
              </span>
            </button>
            
          </div>
          
        </div>
        
      </nav>
     
        <slot></slot>
      
    </div>
    

  </div>
</template>

<script setup>
import { ref, computed } from 'vue';

const links = ref([
  { id: 1, url: "/", label: "Dashboard", iconClasses: "flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white", iconPath: "M11.3 3.3a1 1 0 0 1 1.4 0l6 6 2 2a1 1 0 0 1-1.4 1.4l-.3-.3V19a2 2 0 0 1-2 2h-3a1 1 0 0 1-1-1v-3h-2v3c0 .6-.4 1-1 1H7a2 2 0 0 1-2-2v-6.6l-.3.3a1 1 0 0 1-1.4-1.4l2-2 6-6Z" },
  { id: 2, url: "/index", label: "Visualization", iconClasses: "flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white", iconPath: "M6.143 0H1.857A1.857 1.857 0 0 0 0 1.857v4.286C0 7.169.831 8.00 1.857 8.00h4.286A1.857 1.857 0 0 0 8.00 6.143V1.857A1.857 1.857 0 0 0 6.143 0Zm10 0h-4.286A1.857 1.857 0 0 0 10 1.857v4.286C10 7.169 10.831 8.00 11.857 8.00h4.286A1.857 1.857 0 0 0 18 6.143V1.857A1.857 1.857 0 0 0 16.143 0Zm-10 10H1.857A1.857 1.857 0 0 0 0 11.857v4.286C0 17.169.831 18.00 1.857 18.00h4.286A1.857 1.857 0 0 0 8.00 16.143v-4.286A1.857 1.857 0 0 0 6.143 10Zm10 0h-4.286A1.857 1.857 0 0 0 10 11.857v4.286c0 1.026.831 1.857 1.857 1.857h4.286A1.857 1.857 0 0 0 18 16.143v-4.286A1.857 1.857 0 0 0 16.143 10Z" },
  { id: 3, url: "/crosstabs", label: "Crosstabs Calculation", iconClasses: "flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white", iconPath: "M17.418 3.623A1 1 0 0 0 17 3H2a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h15a1 1 0 0 0 .418-.077l4-2A1 1 0 0 0 22 17V5a1 1 0 0 0-1.582-.814l-4-2Z" },
  { id: 4, url: "/Ttest", label: "T-test Calculation", iconClasses: "flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white", iconPath: "M14 2a3.963 3.963 0 0 0-1.4.267 6.439 6.439 0 0 1-1.331 6.638A4 4 0 1 0 14 2Zm1 9h-1.264A6.957 6.957 0 0 1 15 15v2a2.97 2.97 0 0 1-.184 1H19a1 1 0 0 0 1-1v-1a5.006 5.006 0 0 0-5-5ZM6.5 9a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9ZM8 10H5a5.006 5.006 0 0 0-5 5v2a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1v-2a5.006 5.006 0 0 0-5-5Z" },
  { id: 5, url: "/correlation", label: "Correlation", iconClasses: "flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white", iconPath: "M17 5.923A1 1 0 0 0 16 5h-3V4a4 4 0 1 0-8 0v1H2a1 1 0 0 0-1 .923L.086 17.846A2 2 0 0 0 2.08 20h13.84a2 2 0 0 0 1.994-2.153L17 5.923ZM7 9a1 1 0 0 1-2 0V7h2v2Zm0-5a2 2 0 1 1 4 0v1H7V4Zm6 5a1 1 0 1 1-2 0V7h2v2Z" },
  { id: 6, url: "/anova", label: "Anova", iconClasses: "flex-shrink-0 w-5 h-5 text-gray-500 transition duration-75 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white", iconPath: "M1 8h11m0 0L8 4m4 4-4 4m4-11h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-3" }
]);

const isMenuOpen = ref(false);

const sidebarClasses = computed(() => ({
  'left-0': true,
  'z-40': true,
  'w-64': true,
  'h-screen': true,
  'pt-20': true,
  'transition-transform': true,
  '-translate-x-full': !isMenuOpen.value,
  'bg-white': true,
  'border-r': true,
  'border-gray-200': true,
  'sm:translate-x-0': isMenuOpen.value,
  'dark:bg-gray-800': true,
  'dark:border-gray-700': true
}));

const mainContentClasses = computed(() => ({
  'sm:ml-64': isMenuOpen.value,
  'bg-purple-50': true,
  'h-[100vmin]': true
}));

const toggleMenu = () => {
  isMenuOpen.value = !isMenuOpen.value;
};
</script>

<style scoped>
.pt-20 {
  padding-top: 5px;
}

.hamburger {
  padding: 15px;
  background: none;
  border: none;
  cursor: pointer;
}

.hamburger-box {
  width: 30px;
  height: 24px;
  display: inline-block;
  position: relative;
}

.hamburger-inner {
  width: 100%;
  height: 2px;
  background-color: #fff;
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  transition: background-color 0.3s ease;
}

.hamburger-inner::before,
.hamburger-inner::after {
  content: '';
  width: 100%;
  height: 2px;
  background-color: #fff;
  position: absolute;
  left: 0;
  transition: transform 0.3s ease, background-color 0.3s ease;
}

.hamburger-inner::before {
  top: -8px;
}

.hamburger-inner::after {
  bottom: -8px;
}
</style>
