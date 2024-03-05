<!-- Sidebar.vue -->
<template>
    <div :class="{'hidden': !showSidebar}" class="bg-gray-800 text-white w-64">
      <!-- Sidebar content goes here -->
    </div>
  </template>
  
  <script>
  export default {
    props: ['showSidebar']
  }
  </script>
  