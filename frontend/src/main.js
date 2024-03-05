import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './index.css'



// Import Bootstrap CSS
// import 'bootstrap/dist/css/bootstrap.min.css';



// import 'bootstrap/dist/css/bootstrap.css'
// import bootstrap from 'bootstrap/dist/js/bootstrap.bundle.js'

createApp(App).use(router).mount('#app')
