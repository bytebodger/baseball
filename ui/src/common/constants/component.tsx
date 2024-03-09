import { lazy } from 'react';

const Data = lazy(async () => import('../../pages/data/Data'));
const Home = lazy(async () => import('../../pages/home/Home'));
const Schedules = lazy(async () => import('../../pages/data/schedules/Schedules'));

export const component = {
   data: <Data/>,
   home: <Home/>,
   schedules: <Schedules/>,
}