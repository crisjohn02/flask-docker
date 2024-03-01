import { createRouter, createWebHistory } from 'vue-router'
import DataAnalysis from '../components/DataAnalysis.vue'
import layout from '../components/layout.vue'

const routes = [
  {
    path: '/DataAnalysis',
    name: 'DataAnalysis',
    component: DataAnalysis,
  },
  {
    path: '/layout',
    name: 'Layout',
    component: layout,
  }
  
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
