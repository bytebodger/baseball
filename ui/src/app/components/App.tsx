import dayjs from 'dayjs';
import isSameOrAfter from 'dayjs/plugin/isSameOrAfter';
import isSameOrBefore from 'dayjs/plugin/isSameOrBefore';
import utc from 'dayjs/plugin/utc';
import { useMemo } from 'react';
import { SiteTemplate } from './SiteTemplate';

export const App = () => {
   useMemo(() => {
      dayjs.extend(isSameOrAfter);
      dayjs.extend(isSameOrBefore);
      dayjs.extend(utc);
   }, [])

   return <SiteTemplate/>
}