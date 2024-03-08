import { lazy } from 'react';

const Home = lazy(async () => import('../../pages/home/Home'));

export const component = {
   home: <Home/>,
}