import { createRouter, createWebHistory } from 'vue-router'
import DataAnalysis from '../components/DataAnalysis.vue'
import layout from '../components/layout.vue'
import dashboard from '../components/dashboard.vue'
import index from '../components/index.vue'
import crosstabs from '../components/crosstabs.vue'

const routes = [
  {
    path: '/DataAnalysis',
    name: 'DataAnalysis',
    component: DataAnalysis,
  },
  {
    path: '/',
    name: 'Layout',
    component: layout,
  },
  {
    path: '/index',
    name: 'Index',
    component: index,
  },
  {
    path: '/crosstabs',
    name: 'Crosstab',
    component: crosstabs,
  },
  {
    path: '/dashboard',
    component: dashboard,
  }
  
      
    

  
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
